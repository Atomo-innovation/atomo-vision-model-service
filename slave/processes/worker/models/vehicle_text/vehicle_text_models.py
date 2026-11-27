"""Builds the vehicle detector + scene-text reader for one vehicle_text job.

Shared by onnx/vehicle_text_onnx_detector.py and nb/vehicle_text_nb_detector.py —
they differ only in the runner *specs* (onnxruntime vs. asnn, see
config/models/vehicle_text.yaml's onnx:/nb: blocks); turning those specs into
runners, and the runners into a VehicleDetector + PaddleSceneOCR, is the same.
Replaces EdgeOCR's app.py load_runners()/make_paddle_ocr().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from slave.logging.logger import get_logger
from slave.processes.worker.models.vehicle_text.ocr_runtime import make_runner
from slave.processes.worker.models.vehicle_text.ppocr_reader import PaddleSceneOCR, load_charset
from slave.processes.worker.models.vehicle_text.vehicle_detector import VehicleDetector

log = get_logger(__name__)

COCO_CLASSES = 80          # YOLO11n is the stock COCO model
YOLO_STRIDES = (8, 16, 32)


def yolo_output_shape(imgsz: int) -> List[List[int]]:
    """(1, 4 + 80, anchors) — NPU runners need fixed output shapes up front."""
    anchors = sum((imgsz // s) ** 2 for s in YOLO_STRIDES)
    return [[1, 4 + COCO_CLASSES, anchors]]


def build_runners(
    ext_dir: str,
    yolo_spec: dict,
    det_spec: dict,
    rec_specs: Dict[str, dict],
    model_config: dict,
    threads: int | None,
) -> dict:
    """One runner per network. Each spec is an ocr_runtime.make_runner() entry;
    its files (``file``, ``nb``, ``library``, ``tail``) resolve against ext_dir.
    """
    folder = Path(ext_dir)
    imgsz = int(model_config["vehicle"]["imgsz"])
    det_size = int(model_config["ocr"]["det_size"])
    networks = [
        ("yolo", yolo_spec, yolo_output_shape(imgsz)),
        ("det", det_spec, [[1, 1, det_size, det_size]]),
        *((script, spec, None) for script, spec in rec_specs.items()),
    ]
    runners: dict = {}
    try:
        for name, spec, shapes in networks:
            runners[name] = make_runner(spec, folder, spec.get("file", ""), threads, shapes)
    except Exception:
        # Don't leave the graphs that did load holding the NPU (the watchdog retries).
        release_runners(runners)
        raise
    for name, runner in runners.items():
        log.info("vehicle_text %-9s %s", name, runner.name)
    return runners


def build_vehicle_detector(runner, model_config: dict) -> VehicleDetector:
    vehicle = model_config["vehicle"]
    classes = {int(k): str(v) for k, v in vehicle["classes"].items()}
    # Low-score boxes are kept too: ByteTrack's second matching pass needs them.
    min_score = min(float(model_config["tracker"]["low_thresh"]), float(vehicle["confidence"]))
    return VehicleDetector(runner, int(vehicle["imgsz"]), classes, min_score, float(vehicle.get("nms_iou", 0.7)))


def build_scene_ocr(runners: dict, model_config: dict, rec_width: int, root_dir: str) -> PaddleSceneOCR:
    scripts = list(model_config["languages"])
    charsets = {
        s: load_charset(Path(root_dir) / model_config["recognizers"][s]["dict"]) for s in scripts
    }
    ocr_cfg = dict(model_config["ocr"])
    ocr_cfg["rec_width"] = int(rec_width)
    language_id = dict(model_config.get("language_id") or {})
    if language_id.get("enabled"):
        language_id["model"] = os.path.join(root_dir, language_id["model"])
    ocr_cfg["language_id"] = language_id
    return PaddleSceneOCR({"det": runners["det"], **{s: runners[s] for s in scripts}}, charsets, ocr_cfg)


def release_runners(runners: dict) -> None:
    """Free NPU graphs; onnxruntime sessions just go out of scope.

    EdgeOCR never releases (one process per run). The slave restarts workers on
    job changes, so NPU handles are released the same way runtimes/npu/nb_runtime.py
    does it — whichever release call this asnn build exposes.
    """
    for name, runner in runners.items():
        npu = getattr(runner, "npu", runner)   # YoloDflRunner / TailRunner wrap the NPU runner
        net = getattr(npu, "net", None)
        if net is None:
            continue
        release_fn = (getattr(net, "release", None) or getattr(net, "nn_release", None)
                      or getattr(net, "nn_deinit", None))
        try:
            if release_fn:
                release_fn()
        except Exception:
            log.exception("error releasing NPU runner %s", name)
    runners.clear()
