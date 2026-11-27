"""Vehicle Text `.onnx` detector — every network on CPU (onnxruntime).

Loads YOLO11n (the job's model file, onnx/vehicle_text.onnx), the PP-OCR
text-line detector and one CTC recognizer per configured script, all from
onnx/, per config/models/vehicle_text.yaml's `onnx:` block (EdgeOCR profile
cpu_quality_ppocr). Holds no per-frame state: tracking/OCR scheduling/voting
live in the worker.
"""

from __future__ import annotations

import os

import numpy as np

from slave.processes.worker.models.vehicle_text.vehicle_text_models import (
    build_runners,
    build_scene_ocr,
    build_vehicle_detector,
    release_runners,
)


class VehicleTextOnnxDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        block = model_config["onnx"]
        ext_dir = os.path.dirname(model_path)      # .../vehicle_text/onnx
        root_dir = os.path.dirname(ext_dir)        # .../vehicle_text (dictionaries)
        threads = block.get("threads")
        onnx = {"backend": "onnxruntime"}
        self._runners = build_runners(
            ext_dir,
            yolo_spec={**onnx, "file": os.path.basename(model_path)},
            det_spec={**onnx, "file": block["det"]},
            rec_specs={s: {**onnx, "file": block["recognizers"][s]} for s in model_config["languages"]},
            model_config=model_config,
            threads=threads,
        )
        try:
            self._vehicles = build_vehicle_detector(self._runners["yolo"], model_config)
            self.ocr = build_scene_ocr(self._runners, model_config, block["rec_width"], root_dir)
            self.detect_every = max(1, int(block.get("detect_every", 1)))
            self._vehicles.detect(np.zeros((720, 1280, 3), np.uint8))  # warm-up, as EdgeOCR does
        except Exception:
            release_runners(self._runners)
            raise

    def detect(self, frame: np.ndarray):
        """(boxes N x 4 xyxy frame pixels, scores N, labels N) — vehicle_tracker's input."""
        return self._vehicles.detect(frame)

    def release(self) -> None:
        release_runners(self._runners)
