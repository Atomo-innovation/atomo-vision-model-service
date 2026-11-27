"""Standalone Vehicle Color `.so` worker."""

from __future__ import annotations

from slave.processes.worker.models.vehicle_color.nb.vehicle_color_nb_worker import (
    VehicleColorNbWorker,
)

# .so path shares the same runtime frame loop as .nb
VehicleColorSoWorker = VehicleColorNbWorker
