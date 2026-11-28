"""Which jobs run the vehicle pipeline, and its parts for one runtime.

Everything here only *builds* things (detector, colour classifier, OCR,
plate detector); the per-frame logic lives in vehicle_pipeline_worker.py.
Each builder imports its runtime lazily, so a .nb/.so board never needs
torch and a PC never needs asnn. See config/models/vehicle_pipeline.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import yaml

from slave.models.job import JobConfig, ModelAssignment

PIPELINE_CONFIG_FILE = "vehicle_pipeline.yaml"
# Models the pipeline takes over, and the ones that make a job use it (a
# number_plate-only job has no vehicle detector, so it keeps its own worker).
VEHICLE_MODEL_TYPES = ("vehicle", "vehicle_color", "vehicle_text", "number_plate")
PIPELINE_TRIGGER_TYPES = ("vehicle", "vehicle_color", "vehicle_text")
RUNTIMES = ("pt", "onnx", "nb", "so")


def load_yaml(config_root: str, name: str) -> dict:
    path = os.path.join(config_root, "models", name)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def pipeline_assignments(job: JobConfig, pipeline_cfg: dict) -> List[ModelAssignment]:
    """The job's assignments the pipeline runs, or [] when it doesn't apply."""
    if not pipeline_cfg.get("enabled", False):
        return []
    group = [a for a in job.model_assignments if a.model_type in VEHICLE_MODEL_TYPES]
    if not any(a.model_type in PIPELINE_TRIGGER_TYPES for a in group):
        return []
    return group


def choose_runtime(assignments: List[ModelAssignment], pipeline_cfg: dict) -> str:
    """First of runtime_priority used by any assignment's model_name extension."""
    used = {a.model_extension.lstrip(".").lower() for a in assignments}
    for runtime in pipeline_cfg.get("runtime_priority", ("nb", "so", "pt", "onnx")):
        if runtime in used and runtime in RUNTIMES:
            return runtime
    raise ValueError(f"no supported runtime in {sorted(used)} (supported: {RUNTIMES})")


def build_vehicle_detector(runtime: str, file_name: str, models_root: str, vehicle_cfg: dict) -> Tuple[object, float]:
    """(detector, min score of a real sighting). All 9 classes come back —
    violations are read from no-seatbelt/no_helmet, so nothing is ignored.

    The detector itself keeps boxes down to tracker.low_thresh, like
    vehicle_text/vehicle_color's YOLO: ByteTrack's second pass needs them to
    keep a vehicle on one track_id (a vehicle split over several track_ids
    splits its OCR votes, and its text never gets read twice). Boxes below
    the returned score only keep a track alive.
    """
    folder = os.path.join(models_root, "vehicle")
    path = os.path.join(folder, file_name)
    thresholds = dict(vehicle_cfg.get("thresholds", {}))
    nb_cfg = dict(vehicle_cfg.get("nb", {}))
    low = float(vehicle_cfg.get("tracker", {}).get("low_thresh", 0.1))
    if runtime in ("pt", "onnx"):
        min_score = float(thresholds.get("obj_thresh", 0.25))
        thresholds["obj_thresh"] = min(low, min_score)
    else:
        min_score = float(nb_cfg.get("obj_thresh", thresholds.get("obj_thresh", 0.25)))
        nb_cfg["obj_thresh"] = min(low, min_score)
    cfg = {**vehicle_cfg, "ignore_classes": [], "thresholds": thresholds, "nb": nb_cfg}
    if runtime in ("pt", "onnx"):
        # Ultralytics loads both the .pt and its .onnx export (onnxruntime).
        from slave.processes.worker.models.vehicle.pt.vehicle_pt_detector import VehiclePtDetector
        return VehiclePtDetector(path, cfg, file_name), min_score
    if runtime == "nb":
        from slave.processes.worker.models.vehicle.nb.vehicle_nb_detector import VehicleNbDetector
        library = cfg.get("nb", {}).get("library_file", "")
        return VehicleNbDetector(path, os.path.join(folder, library) if library else None, cfg, file_name), min_score
    from slave.processes.worker.models.vehicle.so.vehicle_so_detector import VehicleSoDetector
    return VehicleSoDetector(path, cfg, file_name), min_score


def build_color_classifier(spec: dict, models_root: str, vehicle_cfg: dict):
    from slave.processes.worker.models.vehicle_color.color_classifier import VehicleColorClassifier

    folder = os.path.join(models_root, "vehicle_color")
    nb_cfg = vehicle_cfg.get("nb", {})
    library = spec.get("library")
    return VehicleColorClassifier(
        os.path.join(folder, spec["model"]),
        device=str(vehicle_cfg.get("color", {}).get("device", "auto")),
        library_path=os.path.join(folder, library) if library else None,
        level=int(nb_cfg.get("level", 0)),
        device_id=int(nb_cfg.get("device_id", 0)),
        board=str(nb_cfg.get("board", "Electron")),
    )


def build_ocr(block_name: str, models_root: str, text_cfg: dict):
    """(PP-OCR reader, its runners) from vehicle_text.yaml's onnx:/nb: block —
    only the text detector + recognizers; vehicles come from the pipeline."""
    from slave.processes.worker.models.vehicle_text.ocr_runtime import make_runner
    from slave.processes.worker.models.vehicle_text.vehicle_text_models import build_scene_ocr, release_runners

    block = text_cfg[block_name]
    root = Path(models_root) / "vehicle_text"
    folder = root / block_name
    threads = block.get("threads")
    if block_name == "onnx":
        onnx = {"backend": "onnxruntime"}
        det_spec = {**onnx, "file": block["det"]}
        rec_specs = {s: {**onnx, "file": block["recognizers"][s]} for s in text_cfg["languages"]}
    else:
        det_spec = dict(block["det"])
        rec_specs = {s: dict(block["recognizers"][s]) for s in text_cfg["languages"]}
    det_size = int(text_cfg["ocr"]["det_size"])
    runners: dict = {}
    try:
        runners["det"] = make_runner(det_spec, folder, det_spec.get("file", ""), threads, [[1, 1, det_size, det_size]])
        for script, spec in rec_specs.items():
            runners[script] = make_runner(spec, folder, spec.get("file", ""), threads, None)
        return build_scene_ocr(runners, text_cfg, int(block["rec_width"]), str(root)), runners
    except Exception:
        release_runners(runners)  # free NPU graphs that did load before the error propagates
        raise


def build_plate_detector(runtime: str, file_name: str, models_root: str, plate_cfg: dict):
    folder = os.path.join(models_root, "number_plate")
    path = os.path.join(folder, file_name)
    if runtime in ("pt", "onnx"):
        from slave.processes.worker.models.number_plate.pt.number_plate_pt_detector import NumberPlatePtDetector
        return NumberPlatePtDetector(path, plate_cfg, file_name)
    if runtime == "nb":
        from slave.processes.worker.models.number_plate.nb.number_plate_nb_detector import NumberPlateNbDetector
        library = plate_cfg.get("nb", {}).get("library_file", "")
        return NumberPlateNbDetector(path, os.path.join(folder, library) if library else None, plate_cfg, file_name)
    from slave.processes.worker.models.number_plate.so.number_plate_so_detector import NumberPlateSoDetector
    return NumberPlateSoDetector(path, plate_cfg, file_name)
