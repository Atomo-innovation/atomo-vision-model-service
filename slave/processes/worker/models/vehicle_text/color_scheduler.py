"""Asynchronous, dynamically toggleable Vehicle Color scheduler for vehicle_text.

Shares the exact tracked vehicle boxes produced by vehicle_text's single
VehicleDetector + ByteTracker, classifying color on a dedicated background
thread so neither the main frame loop nor PaddleSceneOCR is ever blocked.

Supports:
- Dynamic ON/OFF toggling (via `set_enabled()` or polling `vehicle_text.yaml`
  mtime without restarting the service).
- Zero overhead when OFF (thread stopped, classifier weights released).
- CPU (.pt / .onnx via VehicleColorClassifier) and Amlogic A311D NPU (.nb / .so
  via ASNN with output_tensor=1).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from slave.logging.logger import get_logger
from slave.processes.worker.models.vehicle_color.color_classifier import (
    DEFAULT_COLOR_NAMES,
    VehicleColorClassifier,
)
from slave.processes.worker.models.vehicle_text.vehicle_tracker import Track
from slave.runtimes.common import runtime_utils as ru

log = get_logger(__name__)


@dataclass
class _ColorTrackState:
    attempts: int = 0
    pending: bool = False
    last_area: int = 0
    votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    best_conf: Dict[str, float] = field(default_factory=dict)


@dataclass
class _ColorJob:
    track_id: int
    crop: np.ndarray


class _NpuColorBackend:
    """Direct A311D ASNN single-output classifier backend for .nb/.so models."""

    def __init__(
        self, model_path: str, input_size: Tuple[int, int] = (224, 224), library_path: Optional[str] = None
    ) -> None:
        from asnn.api import asnn
        from asnn.types import asnn_stat, output_format

        if not library_path:
            dir_name = os.path.dirname(model_path)
            lib_candidates = [
                os.path.join(dir_name, "libnn_vehicle_color.so"),
                os.path.join(dir_name, "vehicle_color.so"),
                os.path.splitext(model_path)[0] + ".so",
            ]
            library_path = next((p for p in lib_candidates if os.path.isfile(p)), None)

        self._net = asnn("Electron")
        status = self._net.nn_init(library=library_path, model=model_path, level=0)
        if status != asnn_stat.STAT_SUCCESS:
            raise RuntimeError(f"ASNN could not initialize color model {model_path}")
        self._output_format = output_format.OUT_FORMAT_FLOAT32
        self._input_size = input_size
        self._class_names = DEFAULT_COLOR_NAMES

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        resized = cv2.resize(crop, self._input_size)
        tensor = ru.to_input_tensor(resized)
        outputs = self._net.nn_inference(
            [tensor],
            platform="ONNX",
            reorder="2 1 0",
            input_tensor=1,
            output_tensor=1,
            output_format=self._output_format,
        )
        logits = np.asarray(outputs[0], dtype=np.float32).flatten()
        if np.all(logits >= 0) and abs(float(np.sum(logits)) - 1.0) < 1e-2:
            probs = logits
        else:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        top1_idx = int(np.argmax(probs))
        return self._class_names.get(top1_idx, str(top1_idx)), float(probs[top1_idx])

    def release(self) -> None:
        if self._net is not None:
            release_fn = (
                getattr(self._net, "release", None)
                or getattr(self._net, "nn_release", None)
                or getattr(self._net, "nn_deinit", None)
            )
            if release_fn:
                release_fn()
            self._net = None


SUPPORTED_COLOR_EXTENSIONS = (".pt", ".onnx", ".nb", ".so")


def _resolve_color_model(color_cfg: dict, model_extension: str) -> Tuple[str, Optional[str]]:
    """(color model file, its .so library or None) for the active vehicle_text job type.

    vehicle_text.yaml's color: block names the files per job type (color.onnx /
    color.nb; a .so job uses the nb entry — the .so is only the .nb's library).
    Without such an entry, falls back to color.model_name / vehicle_color<ext>.
    """
    ext = model_extension.lower()
    if ext not in SUPPORTED_COLOR_EXTENSIONS:
        raise ValueError(
            f"Unsupported vehicle_text extension '{model_extension}' for auto color selection "
            f"(supported: {SUPPORTED_COLOR_EXTENSIONS})"
        )
    key = ext.lstrip(".")
    entry = color_cfg.get(key) or (color_cfg.get("nb") if key == "so" else None)
    if isinstance(entry, dict) and entry.get("model"):
        return str(entry["model"]), entry.get("library")
    if isinstance(entry, str) and entry:
        return entry, None
    name = (color_cfg.get("model_name") or "").strip()
    if not name or name.lower() == "auto":
        return f"vehicle_color{ext}", None
    if not os.path.splitext(name)[1]:
        return f"{name}{ext}", None
    return name, None


def _resolve_color_model_path(models_dir: str, model_name: str) -> Optional[str]:
    candidates = [
        model_name if os.path.isabs(model_name) else "",
        os.path.join(models_dir, model_name),
        os.path.join(models_dir, "..", "vehicle_color", model_name),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def _load_classifier(models_dir: str, model_name: str, device: str, library: Optional[str] = None):
    model_path = _resolve_color_model_path(models_dir, model_name)
    if not model_path:
        raise FileNotFoundError(
            f"Vehicle color model '{model_name}' not found under {models_dir} or sibling vehicle_color/"
        )
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in SUPPORTED_COLOR_EXTENSIONS:
        raise ValueError(f"Unsupported color model extension '{ext}' for {model_path}")
    if ext in (".nb", ".so"):
        # The library sits next to its .nb model.
        library_path = os.path.join(os.path.dirname(model_path), library) if library else None
        return _NpuColorBackend(model_path, library_path=library_path)
    return VehicleColorClassifier(model_path, device=device)


class ColorScheduler:
    """Non-blocking per-track vehicle color classifier with dynamic ON/OFF toggle."""

    def __init__(
        self,
        models_dir: str,
        color_cfg: dict,
        model_extension: str = ".onnx",
        config_path: Optional[str] = None,
    ) -> None:
        self._models_dir = models_dir
        self._model_extension = model_extension.lower()
        self._config_path = config_path
        self._last_mtime: Optional[float] = self._stat_mtime(config_path)
        self._last_poll = time.monotonic()

        self._lock = threading.Lock()
        self._classifier = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._states: Dict[int, _ColorTrackState] = {}
        self._jobs: queue.Queue[_ColorJob] = queue.Queue(maxsize=max(1, int(color_cfg.get("queue_size", 8))))

        self._apply_config(color_cfg)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def start(self) -> None:
        with self._lock:
            if self._enabled:
                self._ensure_started_locked()

    def stop(self) -> None:
        with self._lock:
            self._disable_locked()

    def set_enabled(self, enabled: bool) -> None:
        """Dynamically enable or disable vehicle color classification at runtime."""
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = bool(enabled)
            if self._enabled:
                self._ensure_started_locked()
            else:
                self._disable_locked()

    def maybe_reload_config(self) -> None:
        """Check vehicle_text.yaml mtime and apply toggle/settings dynamically."""
        if not self._config_path:
            return
        now = time.monotonic()
        if now - self._last_poll < self._poll_interval:
            return
        self._last_poll = now
        mtime = self._stat_mtime(self._config_path)
        if mtime is None or mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            color_cfg = data.get("color") or {}
            with self._lock:
                self._apply_config(color_cfg)
        except Exception:
            log.exception("failed to hot-reload color config from %s", self._config_path)

    def observe(self, tracks: List[Track], frame: np.ndarray) -> None:
        """Queue color jobs for tracked vehicles without blocking the caller."""
        if not self.enabled or not tracks:
            return
        fh, fw = frame.shape[:2]
        for track in tracks:
            x1, y1, x2, y2 = track.box
            if (x2 - x1) < self._min_crop_pixels or (y2 - y1) < self._min_crop_pixels:
                continue
            with self._lock:
                if not self._enabled or self._classifier is None:
                    return
                state = self._states.get(track.track_id)
                if state is None:
                    state = _ColorTrackState()
                    self._states[track.track_id] = state
                if state.pending or state.attempts >= self._max_attempts:
                    continue
                if state.attempts > 0 and track.area < state.last_area * self._regrow_ratio:
                    continue
                state.pending = True

            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(fw, x2), min(fh, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                with self._lock:
                    state.pending = False
                continue
            crop = frame[cy1:cy2, cx1:cx2].copy()
            try:
                self._jobs.put_nowait(_ColorJob(track.track_id, crop))
                with self._lock:
                    state.attempts += 1
                    state.last_area = track.area
            except queue.Full:
                with self._lock:
                    state.pending = False

    def get_color(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Return voted (color, confidence) for a track, or None if disabled/unavailable."""
        with self._lock:
            if not self._enabled:
                return None
            state = self._states.get(track_id)
            if state is None or not state.votes:
                return None
            best_color = max(state.votes, key=state.votes.get)
            conf = state.best_conf.get(best_color, 0.0)
            if conf < self._confidence_threshold:
                return None
            return best_color, conf

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._states.pop(track_id, None)

    # ------------------------------------------------------------------ internals
    def _apply_config(self, color_cfg: dict) -> None:
        new_enabled = bool(color_cfg.get("enabled", False))
        try:
            new_model, new_library = _resolve_color_model(color_cfg, self._model_extension)
        except ValueError as exc:
            log.warning("%s; disabling vehicle color detection", exc)
            self._enabled = False
            self._model_name = ""
            self._library = None
            self._device = str(color_cfg.get("device", "auto"))
            self._disable_locked()
            return
        new_device = str(color_cfg.get("device", "auto"))
        self._confidence_threshold = float(color_cfg.get("confidence_threshold", 0.40))
        self._min_crop_pixels = int(color_cfg.get("min_crop_pixels", 40))
        self._poll_interval = max(0.5, float(color_cfg.get("poll_interval_seconds", 2.0)))
        self._max_attempts = max(1, int(color_cfg.get("max_attempts", 3)))
        self._regrow_ratio = float(color_cfg.get("regrow_ratio", 1.4))

        model_changed = (
            getattr(self, "_model_name", None) != new_model
            or getattr(self, "_library", None) != new_library
            or getattr(self, "_device", None) != new_device
        )
        self._model_name = new_model
        self._library = new_library
        self._device = new_device
        self._enabled = new_enabled

        if not self._enabled:
            self._disable_locked()
        else:
            if model_changed and self._classifier is not None:
                self._disable_locked()
                self._enabled = True
            self._ensure_started_locked()

    def _ensure_started_locked(self) -> None:
        if self._classifier is None:
            try:
                self._classifier = _load_classifier(
                    self._models_dir, self._model_name, self._device, self._library
                )
                log.info(
                    "vehicle_text color classifier enabled (model=%s, format=%s)",
                    self._model_name,
                    self._model_extension,
                )
            except (FileNotFoundError, ValueError) as exc:
                log.warning("%s; disabling vehicle color detection", exc)
                self._enabled = False
                self._classifier = None
                return
            except Exception:
                log.exception("failed to load vehicle color model '%s'; disabling color", self._model_name)
                self._enabled = False
                self._classifier = None
                return
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="vehicle-color-worker", daemon=True)
            self._thread.start()

    def _disable_locked(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        classifier = self._classifier
        self._classifier = None
        self._states.clear()
        while True:
            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                break
        if thread is not None and thread.is_alive():
            self._lock.release()
            try:
                thread.join(timeout=5)
            finally:
                self._lock.acquire()
        if classifier is not None:
            try:
                classifier.release()
            except Exception:
                log.exception("error releasing vehicle color classifier")
            log.info("vehicle_text color classifier disabled and released")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            color_res: Optional[Tuple[str, float]] = None
            try:
                with self._lock:
                    clf = self._classifier
                if clf is not None:
                    color_res = clf.classify(job.crop)
            except Exception:
                log.exception("vehicle color classification failed for track_id=%s", job.track_id)
            finally:
                with self._lock:
                    state = self._states.get(job.track_id)
                    if state is not None:
                        state.pending = False
                        if color_res is not None:
                            color, conf = color_res
                            if color and color != "unknown":
                                state.votes[color] += conf
                                state.best_conf[color] = max(state.best_conf.get(color, 0.0), conf)
                self._jobs.task_done()

    @staticmethod
    def _stat_mtime(path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        try:
            return os.stat(path).st_mtime
        except OSError:
            return None
