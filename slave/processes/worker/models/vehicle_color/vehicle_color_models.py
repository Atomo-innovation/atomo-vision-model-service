"""Builds the vehicle detector + color classifier for one vehicle_color job.

Shared by onnx/, nb/ and so/vehicle_color_*_detector.py — they differ only
in which config/models/vehicle_color.yaml block (onnx:/nb:/so:) names the
files; turning that block into a YOLO runner + VehicleColorClassifier, and
a frame into colored vehicle Detections, is the same.

The vehicle detector is vehicle_text's own (vehicle_text/vehicle_detector.py
+ ocr_runtime.make_runner), run on vehicle_color's own copy of the YOLO11n
files — no OCR. All files resolve flat against models_dir
(processes/worker/models/vehicle_color/).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from slave.processes.worker.models.vehicle_color.color_classifier import VehicleColorClassifier
from slave.processes.worker.models.vehicle_color.vehicle_color_events import agnostic_nms
from slave.processes.worker.models.vehicle_text.ocr_runtime import make_runner
from slave.processes.worker.models.vehicle_text.vehicle_detector import VehicleDetector
from slave.processes.worker.models.vehicle_text.vehicle_text_models import (
    release_runners,
    yolo_output_shape,
)


def build_yolo_runner(models_dir: str, block: dict, model_config: dict):
    """onnxruntime runner for a file name, or an NPU runner for an asnn spec."""
    yolo = block["yolo"]
    if isinstance(yolo, str):
        return make_runner({"backend": "onnxruntime"}, Path(models_dir), yolo, block.get("threads"))
    imgsz = int(model_config["vehicle"]["imgsz"])
    return make_runner(dict(yolo), Path(models_dir), "", block.get("threads"), yolo_output_shape(imgsz))


def build_vehicle_detector(runner, model_config: dict) -> VehicleDetector:
    vehicle = model_config["vehicle"]
    classes = {int(k): str(v) for k, v in vehicle["classes"].items()}
    # Low-score boxes are kept too: ByteTrack's second matching pass needs them.
    min_score = min(float(model_config["tracker"]["low_thresh"]), float(vehicle["confidence"]))
    return VehicleDetector(runner, int(vehicle["imgsz"]), classes, min_score, float(vehicle.get("nms_iou", 0.7)))


def build_color_classifier(models_dir: str, block: dict, model_config: dict) -> VehicleColorClassifier:
    """The color model named in block["color"]; its runner follows the file's extension."""
    spec = block["color"]
    color_cfg = model_config.get("color", {})
    size = int(color_cfg.get("input_size", 224))
    library = spec.get("library")
    return VehicleColorClassifier(
        os.path.join(models_dir, spec["model"]),
        device=str(spec.get("device", "cpu")),
        class_names=dict(enumerate(color_cfg["classes"])) if color_cfg.get("classes") else None,
        input_size=(size, size),
        library_path=os.path.join(models_dir, library) if library else None,
        level=int(block.get("level", 0)),
        device_id=int(block.get("device_id", 0)),
        board=str(block.get("board", "Electron")),
    )


class VehicleColorDetector:
    """YOLO vehicles (detect) + the color classifier (classify). Holds no
    per-frame state: tracking and color voting live in vehicle_color_events.py.
    """

    def __init__(self, models_dir: str, block: dict, model_config: dict) -> None:
        self._agnostic_iou = float(model_config.get("tracker", {}).get("agnostic_nms_iou", 0.7))
        self._runners = {"yolo": build_yolo_runner(models_dir, block, model_config)}
        try:
            self._vehicles = build_vehicle_detector(self._runners["yolo"], model_config)
            self._classifier = build_color_classifier(models_dir, block, model_config)
            self._vehicles.detect(np.zeros((720, 1280, 3), np.uint8))  # warm-up, as vehicle_text does
        except Exception:
            release_runners(self._runners)  # free the loaded NPU graph before the error propagates
            raise

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """(boxes N x 4 xyxy frame pixels, scores N, labels N) — vehicle_tracker's input."""
        return agnostic_nms(*self._vehicles.detect(frame), self._agnostic_iou)

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """(color name, confidence) of one vehicle crop."""
        return self._classifier.classify(crop)

    def release(self) -> None:
        try:
            self._classifier.release()
        finally:
            release_runners(self._runners)
