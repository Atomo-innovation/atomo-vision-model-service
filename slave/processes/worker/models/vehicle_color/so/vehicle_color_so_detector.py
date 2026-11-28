"""Vehicle Color `.so` detector — Amlogic A311D NPU.

The .so files are the .nb models' companion libraries, not standalone
models, so this runs the same .nb + .so pairs as vehicle_color_nb_detector.py,
named in config/models/vehicle_color.yaml's `so:` block.
"""

from __future__ import annotations

from slave.processes.worker.models.vehicle_color.vehicle_color_models import VehicleColorDetector


class VehicleColorSoDetector(VehicleColorDetector):
    def __init__(self, models_dir: str, model_config: dict, model_name: str) -> None:
        super().__init__(models_dir, model_config["so"], model_config)
