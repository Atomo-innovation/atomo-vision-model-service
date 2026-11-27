"""Decides when a tracked vehicle's reading is final and turns it into one Detection.

Port of EdgeOCR's events.py EventPublisher.poll()/_trusted(): a vehicle is
published once it has left the picture (``lost_seconds``) or been watched
``max_wait_seconds``, carrying the best reading of its whole visit — only the
plate and text lines read the same way in ``min_reads`` frames (or once with
``single_read_confidence``). Unlike EdgeOCR, each vehicle is published exactly
once (no "updated" re-publishes, which would reach the Master as duplicate
detections), and its MinIO frame/NATS message go through the worker's
FrameUploader/DetectionPublisher instead of EdgeOCR's own MinIO client + ffmpeg
clips.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from slave.models.detection import Detection
from slave.processes.worker.models.vehicle_text.ocr_scheduler import OCRScheduler, TrackState
from slave.processes.worker.models.vehicle_text.ocr_types import Line, Word
from slave.processes.worker.models.vehicle_text.plate_parser import parse_plate
from slave.processes.worker.models.vehicle_text.text_overlay import Overlay
from slave.utils.time_utils import IST

CLASS_NAME = "vehicle_text"   # == model_type, so group_detections() finds the job's assignment ids
CLASS_ID = 0
# Annotated frames kept so a vehicle finishing later on an already-published
# frame is drawn onto the same image (see poll()). ~6 MB each at 1080p.
ANNOTATED_FRAMES_KEPT = 8


@dataclass
class FinishedVehicle:
    track_id: int
    frame_id: int                 # frame of the vehicle's best reading
    occurred_at: str              # IST ISO time of that frame
    image: np.ndarray             # that full frame, annotated
    detection: Detection


class VehicleTextEvents:
    def __init__(self, publish_cfg: dict, model_name: str) -> None:
        self._lost_seconds = float(publish_cfg.get("lost_seconds", 2.0))
        self._max_wait = float(publish_cfg.get("max_wait_seconds", 30))
        self._min_reads = max(1, int(publish_cfg.get("min_reads", 2)))
        self._single_read_confidence = float(publish_cfg.get("single_read_confidence", 0.9))
        self._forget_after = float(publish_cfg.get("forget_after_seconds", 120))
        self._model_name = model_name
        self._overlay: Optional[Overlay] = None
        self._annotated: "OrderedDict[int, np.ndarray]" = OrderedDict()  # frame_id -> annotated frame

    def poll(self, scheduler: OCRScheduler, now: float, color_scheduler=None) -> List[FinishedVehicle]:
        """Vehicles whose reading became final since the last poll (``now``: the
        same clock the worker passes to scheduler.observe()).
        """
        ready = []
        with scheduler.lock:
            # Per-OCR-job log, only used by EdgeOCR's CSV output; unbounded on a live stream.
            scheduler.events.clear()
            for track_id, state in list(scheduler.tracks.items()):
                gone = now - state.last_seen
                if gone > self._forget_after and not state.pending:
                    del scheduler.tracks[track_id]   # keeps memory flat on a live stream
                    if color_scheduler is not None:
                        color_scheduler.forget(track_id)
                    continue
                if state.emitted or state.pending:
                    continue
                done = gone >= self._lost_seconds or now - state.first_seen >= self._max_wait
                if not done or state.best is None or state.best_image is None:
                    continue
                plate = scheduler.plate(state)
                if plate is not None and not self._trusted(plate[2], plate[1]):
                    plate = None  # one uncertain read: could be a misread
                lines: List[Tuple[Line, Optional[int]]] = []
                shown: List[Line] = []
                for line in scheduler.text_lines(state):
                    if parse_plate(line.text, 0) is not None:
                        continue  # the voted plate is published instead of raw plate reads
                    count = scheduler.text_reads(state, line)
                    if self._trusted(count, line.confidence):
                        lines.append((line, count))
                        shown.append(line)
                if plate is not None:
                    shown.insert(0, Line([Word(plate[0], "english", plate[1], (0, 0, 0, 0))]))
                if not lines and plate is None:
                    continue  # nothing trustworthy yet; re-checked until forgotten
                state.emitted = True
                image, state.best_image = state.best_image, None  # this vehicle's frame is no longer needed
                color_info = color_scheduler.get_color(track_id) if color_scheduler is not None else None
                ready.append((state, image, plate, lines, shown, color_info))

        finished = []
        for state, image, plate, lines, shown, color_info in ready:
            frame_id = int(state.best_frame)
            # MinIO key is per frame (<worker_key>/frames/frame_<id>.jpg, like every
            # other model), and several vehicles can have their best read on the same
            # frame: they are all drawn onto one image, so re-uploading it keeps every
            # earlier vehicle's box and text on the file their messages point to.
            canvas = self._annotated.get(frame_id)
            if canvas is None:
                canvas = image.copy()  # the raw frame is shared by every vehicle OCR'd on it
                self._annotated[frame_id] = canvas
                while len(self._annotated) > ANNOTATED_FRAMES_KEPT:
                    self._annotated.popitem(last=False)
            if self._overlay is None:
                self._overlay = Overlay(canvas.shape[0])
            tag = f"{state.label} #{state.track_id}"
            if color_info is not None:
                tag += f" ({color_info[0]})"
            self._overlay.draw_vehicle(canvas, state.best_box, tag, shown)
            finished.append(FinishedVehicle(
                track_id=state.track_id,
                frame_id=frame_id,
                occurred_at=datetime.fromtimestamp(state.best_wall_time, IST).isoformat(),
                image=canvas,
                detection=self._detection(state, plate, lines, color_info),
            ))
        return finished

    def _trusted(self, reads: int, confidence: float) -> bool:
        return reads >= self._min_reads or confidence >= self._single_read_confidence

    def _detection(self, state: TrackState, plate, lines, color_info=None) -> Detection:
        ox, oy = state.best_origin  # line boxes are in vehicle-crop pixels

        def to_frame(box) -> List[int]:
            return [int(box[0] + ox), int(box[1] + oy), int(box[2] + ox), int(box[3] + oy)]

        best = state.best
        return Detection(
            class_id=CLASS_ID,
            class_name=CLASS_NAME,
            confidence=float(best.confidence),
            bbox=tuple(float(v) for v in state.best_box),
            model_name=self._model_name,
            extra={
                "vehicle_class": state.label,
                "track_id": state.track_id,
                "plate_number": plate[0] if plate else None,
                "plate_confidence": round(plate[1], 3) if plate else None,
                "plate_reads": plate[2] if plate else None,
                "text": " | ".join(line.text for line, _ in lines) or None,
                "lines": [{
                    "text": line.text,
                    "script": line.script,
                    "confidence": round(line.confidence, 3),
                    "bbox": to_frame(line.box),
                    "reads": reads,
                } for line, reads in lines],
                "language": best.language if best.language != "und" else None,
                "color": color_info[0] if color_info else None,
                "color_confidence": round(color_info[1], 4) if color_info else None,
            },
        )
