"""Vehicle Color Recognition package."""

from slave.processes.worker.models.vehicle_color.color_classifier import (
    DEFAULT_COLOR_NAMES,
    VehicleColorClassifier,
)
from slave.processes.worker.models.vehicle_color.color_enricher import VehicleColorEnricher

__all__ = [
    "DEFAULT_COLOR_NAMES",
    "VehicleColorClassifier",
    "VehicleColorEnricher",
]

