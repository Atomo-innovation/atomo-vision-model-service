"""Vehicle Color `.onnx` detector — every network on CPU (onnxruntime).

YOLO11n vehicles (vehicle_text.onnx) + the color classifier
(vehicle_color.onnx), both named in config/models/vehicle_color.yaml's
`onnx:` block and resolved against processes/worker/models/vehicle_color/.
"""

from __future__ import annotations

from slave.processes.worker.models.vehicle_color.vehicle_color_models import VehicleColorDetector


class VehicleColorOnnxDetector(VehicleColorDetector):
    def __init__(self, models_dir: str, model_config: dict, model_name: str) -> None:
        super().__init__(models_dir, model_config["onnx"], model_config)
