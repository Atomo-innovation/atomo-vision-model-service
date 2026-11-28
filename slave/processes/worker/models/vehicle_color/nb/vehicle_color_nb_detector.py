"""Vehicle Color `.nb` detector — Amlogic A311D NPU.

YOLO11n vehicles (vehicle_text.nb + vehicle_text.so, DFL box decoding in
NumPy — vehicle_text's board-validated asnn runner) + the color classifier
(vehicle_color.nb + libnn_vehicle_color.so, via runtimes/npu/nb_runtime.py),
both named in config/models/vehicle_color.yaml's `nb:` block and resolved
against processes/worker/models/vehicle_color/.
"""

from __future__ import annotations

from slave.processes.worker.models.vehicle_color.vehicle_color_models import VehicleColorDetector


class VehicleColorNbDetector(VehicleColorDetector):
    def __init__(self, models_dir: str, model_config: dict, model_name: str) -> None:
        super().__init__(models_dir, model_config["nb"], model_config)
