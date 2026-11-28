"""Vehicle Color Enricher.

Integrates with Vehicle workers (PT and NB) to post-process detected vehicle
bounding boxes and enrich them with recognized vehicle colors.

Governed dynamically by the model configuration YAML:
    color:
      enabled: true / false
      model_name: vehicle_color.pt
      confidence_threshold: 0.45
      min_crop_pixels: 40

When enabled is False, the classifier is NOT loaded into memory and enrich()
returns the detections unmodified with zero overhead.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.processes.worker.models.vehicle_color.color_classifier import VehicleColorClassifier

log = get_logger(__name__)

# Vehicle classes that should receive color classification
TARGET_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "vehicle"}


class VehicleColorEnricher:
    """Enriches vehicle bounding box detections with color classification."""

    def __init__(self, model_config: dict, models_dir: str) -> None:
        color_cfg = model_config.get("color", {})
        self.enabled = bool(color_cfg.get("enabled", False))

        if not self.enabled:
            self._classifier: Optional[VehicleColorClassifier] = None
            log.info("vehicle color recognition is disabled in config")
            return

        model_name = color_cfg.get("model_name", "vehicle_color.pt")
        device = color_cfg.get("device", "auto")
        self.confidence_threshold = float(color_cfg.get("confidence_threshold", 0.45))
        self.min_crop_pixels = int(color_cfg.get("min_crop_pixels", 40))

        # Look in candidate directories:
        # 1. sibling 'vehicle_color' directory
        # 2. current models_dir
        # 3. repo-relative path
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name),
            os.path.join(models_dir, model_name),
            os.path.join(models_dir, "..", "vehicle_color", model_name),
        ]

        model_path = None
        for cand in candidates:
            cand = os.path.abspath(cand)
            if os.path.isfile(cand):
                model_path = cand
                break

        if not model_path:
            log.warning(
                "vehicle color model '%s' not found in candidates: %s. Disabling color recognition.",
                model_name,
                candidates,
            )
            self.enabled = False
            self._classifier = None
            return

        try:
            log.info("loading vehicle color classifier from: %s", model_path)
            self._classifier = VehicleColorClassifier(model_path, device=device)
            log.info("vehicle color classifier loaded successfully (enabled=True)")
        except Exception:
            log.exception("failed to initialize vehicle color classifier, disabling")
            self.enabled = False
            self._classifier = None

    def enrich(self, frame: np.ndarray, detections: Sequence[Detection]) -> List[Detection]:
        """Classify colors for detected vehicles and add 'color' to Detection.extra."""
        if not self.enabled or self._classifier is None or not detections:
            return list(detections)

        h, w = frame.shape[:2]
        enriched: List[Detection] = []

        for det in detections:
            cls_lower = det.class_name.lower()
            if cls_lower not in TARGET_VEHICLE_CLASSES:
                enriched.append(det)
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
            crop_w = x2 - x1
            crop_h = y2 - y1

            # Reject too-small or out-of-bounds crops
            if crop_w < self.min_crop_pixels or crop_h < self.min_crop_pixels:
                enriched.append(det)
                continue

            clamped_x1 = max(0, min(w - 1, x1))
            clamped_y1 = max(0, min(h - 1, y1))
            clamped_x2 = max(0, min(w, x2))
            clamped_y2 = max(0, min(h, y2))

            crop = frame[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
            if crop.size == 0:
                enriched.append(det)
                continue

            try:
                color, conf = self._classifier.classify(crop)
                if conf >= self.confidence_threshold:
                    det.extra["color"] = color
                    det.extra["color_confidence"] = round(conf, 4)
            except Exception:
                log.exception("failed to classify vehicle crop color for bbox %s", det.bbox)

            enriched.append(det)

        return enriched

    @property
    def classify(self):
        """(color, confidence) of one vehicle crop, or None when color is
        disabled/unavailable — VehicleColorEvents' classify argument."""
        if not self.enabled or self._classifier is None:
            return None
        return self._classifier.classify

    def release(self) -> None:
        if self._classifier is not None:
            self._classifier.release()
            self._classifier = None

