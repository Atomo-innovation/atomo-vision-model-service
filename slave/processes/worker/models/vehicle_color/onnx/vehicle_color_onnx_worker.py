"""Standalone Vehicle Color `.onnx` worker."""

from __future__ import annotations

from slave.processes.worker.models.vehicle_color.pt.vehicle_color_pt_worker import (
    VehicleColorPtWorker,
)

# .onnx worker uses VehicleColorPtWorker pipeline, loading .onnx weights via VehicleColorClassifier
VehicleColorOnnxWorker = VehicleColorPtWorker
