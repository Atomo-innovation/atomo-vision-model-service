"""Vehicle Text `.nb` detector — Amlogic A311D NPU (asnn), int16.

YOLO head (the job's model file, nb/vehicle_text.nb + .so; DFL box decoding in
NumPy), PP-OCR text-line detector and the recognizer backbones run on the NPU;
the recognizers' last layers (nb/*_tail.onnx) run on CPU. Specs come from
config/models/vehicle_text.yaml's `nb:` block, copied from EdgeOCR's
board-validated a311d_npu_ppocr_sqf_all profile — see ocr_runtime.py for why
each NPU setting is what it is.

Uses EdgeOCR's AsnnRunner (ocr_runtime.py), not runtimes/npu/nb_runtime.py:
these graphs need per-model input scale/offset/stretch and fixed output
shapes that NbRuntime doesn't handle.
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


class VehicleTextNbDetector:
    def __init__(self, model_path: str, model_config: dict, model_name: str) -> None:
        block = model_config["nb"]
        ext_dir = os.path.dirname(model_path)      # .../vehicle_text/nb
        root_dir = os.path.dirname(ext_dir)        # .../vehicle_text (dictionaries)
        self._runners = build_runners(
            ext_dir,
            # The job's model_name is the YOLO graph; its .so library comes from the config.
            yolo_spec={**block["yolo"], "nb": os.path.basename(model_path)},
            det_spec=dict(block["det"]),
            rec_specs={s: dict(block["recognizers"][s]) for s in model_config["languages"]},
            model_config=model_config,
            threads=block.get("threads"),
        )
        try:
            self._vehicles = build_vehicle_detector(self._runners["yolo"], model_config)
            self.ocr = build_scene_ocr(self._runners, model_config, block["rec_width"], root_dir)
            self.detect_every = max(1, int(block.get("detect_every", 1)))
            self._vehicles.detect(np.zeros((720, 1280, 3), np.uint8))  # warm-up, as EdgeOCR does
        except Exception:
            release_runners(self._runners)  # free the loaded NPU graphs before the error propagates
            raise

    def detect(self, frame: np.ndarray):
        """(boxes N x 4 xyxy frame pixels, scores N, labels N) — vehicle_tracker's input."""
        return self._vehicles.detect(frame)

    def release(self) -> None:
        release_runners(self._runners)
