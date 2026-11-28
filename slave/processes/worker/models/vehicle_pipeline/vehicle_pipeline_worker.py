"""Vehicle pipeline worker — one worker for all of a job's vehicle models.

    frame -> vehicle detector (ONE run, 9 classes) -> ByteTrack (ONE tracker)
          -> VehicleColorEvents: per-vehicle state, colour votes (vehicle_color),
             best frame, relink, publish-once timing (vehicle.yaml)
          -> vehicle_text:  OCRScheduler reads plate + text of each track
          -> number_plate:  plate boxes + PlateEnricher reads, each plate given
                            to the vehicle box holding its centre
          -> vehicle:       no-seatbelt / no_helmet boxes given to their vehicle
          -> once a vehicle has left (or was watched max_wait_seconds), one
             annotated MinIO frame + one fixed vehicle message for it
             (models/vehicle_event.py), null where a model isn't in the job.

Built by model_manager.py instead of the per-model workers when the job has
any of vehicle / vehicle_color / vehicle_text (config/models/
vehicle_pipeline.yaml). All tracker/vote state is per-job, never shared
(CLAUDE.md §20). No face_controller.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from slave.logging.logger import get_logger
from slave.models.detection import Detection
from slave.models.job import JobConfig, ModelAssignment
from slave.models.vehicle_event import build_vehicle_detection, build_vehicle_message, build_violation
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.vehicle_color.vehicle_color_events import (
    FinishedVehicle,
    VehicleColorEvents,
    agnostic_nms,
    build_tracker,
)
from slave.processes.worker.models.vehicle_pipeline.vehicle_pipeline_parts import (
    build_color_classifier,
    build_ocr,
    build_plate_detector,
    build_vehicle_detector,
    choose_runtime,
)
from slave.processes.worker.models.vehicle_text.ocr_types import Line, Word
from slave.processes.worker.models.vehicle_text.text_overlay import Overlay
from slave.processes.worker.models.vehicle_text.vehicle_tracker import Track
from slave.processes.worker.worker_factory import WorkerIdentity

FRAME_POLL_TIMEOUT_SECONDS = 1.0
# Published frames kept so a vehicle finishing later on an already-published
# frame is drawn onto it with the earlier ones (same as VehicleColorEvents).
ANNOTATED_FRAMES_KEPT = 8
# Plate boxes kept per vehicle (one entry per frame it was seen on), and how
# many frame_ids away from the published frame a plate box may come from —
# further away the vehicle has moved too far for the box to still fit.
PLATE_BOXES_KEPT = 300
PLATE_FRAME_TOLERANCE = 2
PLATE_COLOR = (0, 0, 255)        # BGR red: plate box + number
VIOLATION_COLOR = (0, 0, 255)    # BGR red: violation names
# Same look as frame_annotator.py's vehicle labels.
LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
LABEL_HEIGHT = cv2.getTextSize("Ag", LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)[0][1] + 10
TEXT_LABEL_FRAME_HEIGHT = 720    # Overlay font size for read text: 720 // 36 = 20 px
SPEED_LOG_SECONDS = 10.0
# Same order as every vehicle detector (pt/nb/so) — class_id of the published vehicle.
CLASSES = ("Helmet", "bus", "car", "motorcycle", "no-seatbelt", "no_helmet", "other", "seatbelt", "truck")
COLOR_CLASSES = {"car", "bus", "truck", "motorcycle"}


def _center(box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _draw_label(image: np.ndarray, text: str, x: int, bottom: int, color) -> None:
    """White text on a filled box whose bottom edge is at ``bottom``."""
    (w, h), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    bottom = max(bottom, h + 8)
    cv2.rectangle(image, (x, bottom - h - 8), (x + w + 4, bottom), color, -1)
    cv2.putText(image, text, (x + 2, bottom - 6), LABEL_FONT, LABEL_SCALE, (255, 255, 255), LABEL_THICKNESS)


def _move_with(box, old_outer, new_outer) -> Tuple[float, float, float, float]:
    """box (inside old_outer) put at the same place inside new_outer — a plate
    moved along with its vehicle between two frames close together."""
    ox1, oy1, ox2, oy2 = (float(v) for v in old_outer)
    nx1, ny1, nx2, ny2 = (float(v) for v in new_outer)
    sx, sy = (nx2 - nx1) / max(1.0, ox2 - ox1), (ny2 - ny1) / max(1.0, oy2 - oy1)
    return (nx1 + (box[0] - ox1) * sx, ny1 + (box[1] - oy1) * sy, nx1 + (box[2] - ox1) * sx, ny1 + (box[3] - oy1) * sy)


def _plate_text(text: Optional[str]) -> Optional[str]:
    """Plates are published upper-case without spaces (GJ07AB1234)."""
    return re.sub(r"\s+", "", text).upper() if text else None


@dataclass
class _PlateRead:
    number: Optional[str]
    plate_confidence: Optional[float]
    box_confidence: float
    track_id: int

    def better_than(self, other: Optional["_PlateRead"]) -> bool:
        if other is None:
            return True
        if (self.number is None) != (other.number is None):
            return self.number is not None  # a read plate beats an unread one
        if self.number is not None:
            return (self.plate_confidence or 0.0) > (other.plate_confidence or 0.0)
        return self.box_confidence > other.box_confidence


@dataclass
class _VehicleExtras:
    """What the plate / violation parts found for one vehicle, until it's published."""

    last_update: float
    plate: Optional[_PlateRead] = None
    # frame_id -> {plate track_id: (plate box, its vehicle's box) on that
    # frame}: the published plate bbox is the real box on the published
    # frame, or one from a frame right next to it moved along with the
    # vehicle — never one from far away (see _plate_box()).
    plate_boxes: "OrderedDict[int, Dict[int, Tuple[tuple, tuple]]]" = field(default_factory=OrderedDict)
    # violation_type -> [frames seen, best confidence]
    violations: Dict[str, List[float]] = field(default_factory=dict)


class VehiclePipelineWorker:
    def __init__(
        self,
        job: JobConfig,
        assignments: Sequence[ModelAssignment],
        worker_key: str,
        pipeline_config: dict,
        configs: Dict[str, dict],
        models_root: str,
        detection_publisher: DetectionPublisher,
        frame_uploader: FrameUploader,
        video_manager: VideoManager,
        identity: WorkerIdentity,
    ) -> None:
        self._job = job
        self._worker_key = worker_key
        self._detection_publisher = detection_publisher
        self._frame_uploader = frame_uploader
        self._video_manager = video_manager
        self._identity = identity
        self._log = get_logger(__name__, job_id=job.job_id)

        self._types = {a.model_type for a in assignments}
        self._assign_id = next(
            (a.camera_model_assign_id
             for t in pipeline_config.get("assign_id_priority", ("vehicle", "vehicle_color", "vehicle_text", "number_plate"))
             for a in assignments if a.model_type == t),
            None,
        )
        self.runtime = choose_runtime(list(assignments), pipeline_config)
        files = pipeline_config["runtimes"][self.runtime]
        vehicle_cfg = configs["vehicle"]
        self._vehicle_classes = set(pipeline_config.get("vehicle_classes", ("bus", "car", "motorcycle", "truck", "other")))

        # Built in order, released in reverse on a failure halfway (NPU graphs).
        self._detector = self._classifier = self._scheduler = self._plate_detector = self._plate_enricher = None
        self._ocr_runners: Optional[dict] = None
        try:
            self._detector, min_score = build_vehicle_detector(self.runtime, files["detector"], models_root, vehicle_cfg)
            if "vehicle_color" in self._types:
                self._classifier = build_color_classifier(files["color"], models_root, vehicle_cfg)
            if "vehicle_text" in self._types:
                from slave.processes.worker.models.vehicle_text.ocr_scheduler import OCRScheduler

                text_cfg = configs["vehicle_text"]
                ocr, self._ocr_runners = build_ocr(files["ocr"], models_root, text_cfg)
                self._scheduler = OCRScheduler(ocr, text_cfg["scheduler"], text_cfg["ocr"])
                text_publish = text_cfg.get("publish", {})
                self._min_reads = max(1, int(text_publish.get("min_reads", 2)))
                self._single_read_confidence = float(text_publish.get("single_read_confidence", 0.9))
            if "number_plate" in self._types:
                from slave.processes.worker.models.number_plate.plate_enricher import PlateEnricher

                plate_cfg = configs["number_plate"]
                self._plate_detector = build_plate_detector(self.runtime, files["number_plate"], models_root, plate_cfg)
                self._plate_enricher = PlateEnricher(plate_cfg)
        except Exception:
            self._release_models()
            raise

        violations_cfg = pipeline_config.get("violations", {})
        self._violation_classes = dict(violations_cfg.get("classes", {})) if "vehicle" in self._types else {}
        self._helmet_head_ratio = float(violations_cfg.get("helmet_head_ratio", 0.5))
        self._violation_min_frames = int(violations_cfg.get("min_frames", 2))

        self._min_score = min_score  # lower boxes only keep a track alive, see build_vehicle_detector()
        self._agnostic_iou = float(vehicle_cfg.get("tracker", {}).get("agnostic_nms_iou", 0.7))
        self._forget_after = float(vehicle_cfg.get("publish", {}).get("forget_after_seconds", 120))
        self._tracker = build_tracker(vehicle_cfg, min_score)
        self._events = VehicleColorEvents(
            {**vehicle_cfg, "display_name": "Vehicle"},
            self._classifier.classify if self._classifier is not None else None,
            class_ids={name: i for i, name in enumerate(CLASSES)},
            min_vehicle_score=min_score,
            color_classes=COLOR_CLASSES,
        )
        self._extras: Dict[int, _VehicleExtras] = {}
        # frame_id -> (fields, text lines) of every vehicle published on it, see _annotate().
        self._drawn: "OrderedDict[int, List[Tuple[dict, list]]]" = OrderedDict()
        self._overlay: Optional[Overlay] = None

        self._frame_reader = FrameReader(worker_key, job.camera_stream)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"vehicle-pipeline-{worker_key}", daemon=True)
        self._clock_start = time.monotonic()
        self._published = 0
        self._speed_window_start = time.monotonic()
        self._speed_window_frames = 0

    @property
    def parts(self) -> List[str]:
        return sorted(self._types)

    def start(self) -> None:
        self._frame_reader.start()
        if self._scheduler is not None:
            self._scheduler.start()
        self._thread.start()
        self._log.info("vehicle pipeline started (runtime=%s, models=%s)", self.runtime, ",".join(self.parts))

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("vehicle pipeline stopped (%d vehicle(s) published)", self._published)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
            now = time.monotonic() - self._clock_start
            if frame is None:
                self._publish_finished(now)  # vehicles still finish while the stream is down
                continue
            try:
                self.process_frame(frame.image, frame.frame_id, now, frame.timestamp)
                self._log_speed()
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    # ------------------------------------------------------------ per frame
    def process_frame(self, image: np.ndarray, frame_id: int, now: float, wall_time: float) -> None:
        detections = self._detector.detect(image)
        vehicles = [d for d in detections if d.class_name in self._vehicle_classes]
        boxes = np.array([d.bbox for d in vehicles], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d.confidence for d in vehicles], dtype=np.float32)
        labels = [d.class_name for d in vehicles]
        tracks = self._tracker.update(*agnostic_nms(boxes, scores, labels, self._agnostic_iou))

        self._events.observe(tracks, image, frame_id, now, wall_time)
        if self._scheduler is not None:
            self._scheduler.observe(tracks, image, frame_id, now, wall_time)
        if self._violation_classes:
            self._match_violations(detections, tracks, now)
        if self._plate_detector is not None:
            self._match_plates(image, frame_id, wall_time, tracks, now)
        self._publish_finished(now)

        # Video only records while a vehicle is in frame, with its live track boxes.
        if tracks:
            live = [
                Detection(class_id=-1, class_name=f"{t.label} #{t.track_id}", confidence=t.confidence,
                          bbox=tuple(float(v) for v in t.box), model_name="")
                for t in tracks
            ]
            self._video_manager.feed_frame(draw_detections(image, live))

    def _owner(self, box, tracks: List[Track], labels, grow_up: float = 0.0) -> Optional[Track]:
        """The smallest vehicle box (of these labels) holding box's centre."""
        cx, cy = _center(box)
        best = None
        for track in tracks:
            if track.label not in labels:
                continue
            x1, y1, x2, y2 = track.box
            y1 -= grow_up * (y2 - y1)
            if x1 <= cx <= x2 and y1 <= cy <= y2 and (best is None or track.area < best.area):
                best = track
        return best

    def _extras_for(self, track: Track, now: float) -> _VehicleExtras:
        key = self._events.key_for(track.track_id)
        extras = self._extras.get(key)
        if extras is None:
            extras = self._extras[key] = _VehicleExtras(now)
        extras.last_update = now
        return extras

    def _match_violations(self, detections: List[Detection], tracks: List[Track], now: float) -> None:
        seen: Dict[Tuple[int, str], float] = {}  # (track_id, violation_type) -> best conf on this frame
        for det in detections:
            rule = self._violation_classes.get(det.class_name)
            if rule is None or det.confidence < self._min_score:
                continue
            grow = self._helmet_head_ratio if rule["violation_type"] == "no_helmet" else 0.0
            owner = self._owner(det.bbox, tracks, set(rule["vehicles"]), grow_up=grow)
            if owner is None:
                continue
            k = (owner.track_id, rule["violation_type"])
            seen[k] = max(seen.get(k, 0.0), float(det.confidence))
        by_id = {t.track_id: t for t in tracks}
        for (track_id, violation_type), conf in seen.items():
            entry = self._extras_for(by_id[track_id], now).violations.setdefault(violation_type, [0, 0.0])
            entry[0] += 1  # frames seen
            entry[1] = max(entry[1], conf)

    def _match_plates(self, image: np.ndarray, frame_id: int, wall_time: float, tracks: List[Track], now: float) -> None:
        plates = self._plate_enricher.enrich(image, self._plate_detector.detect(image), frame_id, frame_timestamp=wall_time)
        self._plate_enricher.take_publishable()  # the pipeline publishes per vehicle, not per plate
        for plate, plate_track_id in zip(plates, self._plate_enricher.last_track_ids):
            owner = self._owner(plate.bbox, tracks, self._vehicle_classes)
            if owner is None:
                continue  # a plate outside every vehicle box: no vehicle to give it to
            read = _PlateRead(
                number=_plate_text(plate.extra.get("plate_number")),
                plate_confidence=plate.extra.get("plate_confidence"),
                box_confidence=float(plate.confidence),
                track_id=plate_track_id,
            )
            extras = self._extras_for(owner, now)
            if read.better_than(extras.plate):
                extras.plate = read
            boxes = extras.plate_boxes.setdefault(frame_id, {})
            boxes[plate_track_id] = (tuple(float(v) for v in plate.bbox), tuple(float(v) for v in owner.box))
            while len(extras.plate_boxes) > PLATE_BOXES_KEPT:
                extras.plate_boxes.popitem(last=False)

    @staticmethod
    def _plate_box(extras: Optional[_VehicleExtras], plate: _PlateRead, frame_id: int, vehicle_box) -> Optional[List[float]]:
        """The published plate's box on frame_id (whose vehicle box is
        vehicle_box). If the plate wasn't detected on that very frame, its box
        from the nearest frame within PLATE_FRAME_TOLERANCE is moved with the
        vehicle (same place inside the vehicle box); else None — no bbox, not
        drawn — rather than a box from a frame the vehicle has moved on from."""
        if extras is None:
            return None
        for gap in range(PLATE_FRAME_TOLERANCE + 1):
            for fid in (frame_id - gap, frame_id + gap):
                boxes = extras.plate_boxes.get(fid, {})
                # Its own plate track; or, if the plate tracker re-numbered it,
                # the vehicle's only plate box on that frame.
                found = boxes.get(plate.track_id) or (next(iter(boxes.values())) if len(boxes) == 1 else None)
                if found is None:
                    continue
                box, seen_in = found
                if gap:
                    box = _move_with(box, seen_in, vehicle_box)
                return [round(v, 1) for v in box]
        return None

    # ------------------------------------------------------------ publishing
    def _publish_finished(self, now: float) -> None:
        hold = self._ocr_pending if self._scheduler is not None else None
        for vehicle in self._events.poll(now, hold):
            try:
                self._publish(vehicle)
            except Exception:
                self._log.exception("failed to publish vehicle track_id=%s", vehicle.track_id)
        self._forget(now)

    def _ocr_pending(self, key: int) -> bool:
        with self._scheduler.lock:
            return any(
                state.pending
                for track_id in self._events.track_ids_for(key)
                if (state := self._scheduler.tracks.get(track_id)) is not None
            )

    def _trusted(self, reads: int, confidence: float) -> bool:
        return reads >= self._min_reads or confidence >= self._single_read_confidence

    def _read_text(self, key: int):
        """(plate (number, confidence) or None, [text Lines]) of a vehicle's OCR reads."""
        from slave.processes.worker.models.vehicle_text.plate_parser import parse_plate

        plate, lines, seen = None, [], set()
        with self._scheduler.lock:
            for track_id in self._events.track_ids_for(key):
                state = self._scheduler.tracks.get(track_id)
                if state is None:
                    continue
                read = self._scheduler.plate(state)
                if read is not None and self._trusted(read[2], read[1]) and (plate is None or read[1] > plate[1]):
                    plate = (read[0], read[1])
                for line in self._scheduler.text_lines(state):
                    text = line.text.strip()
                    if not text or text in seen or parse_plate(line.text, 0) is not None:
                        continue  # the plate is published as plate_number, not as text
                    if self._trusted(self._scheduler.text_reads(state, line), line.confidence):
                        seen.add(text)
                        lines.append(line)
        return plate, lines

    def _publish(self, vehicle: FinishedVehicle) -> None:
        key = vehicle.track_id
        det = vehicle.detection
        extras = self._extras.pop(key, None)
        fields = dict(
            vehicle_type=det.class_name,
            vehicle_type_confidence=round(float(det.confidence), 4),
            vehicle_bbox=[round(float(v), 1) for v in det.bbox],
        )
        if self._classifier is not None:
            fields.update(colour=det.extra.get("color"), colour_confidence=det.extra.get("color_confidence"))

        if self._plate_detector is not None:
            plate = extras.plate if extras is not None else None
            fields["plate_confirmed"] = bool(plate is not None and plate.number is not None)
            if plate is not None:
                fields.update(
                    confidence=round(plate.box_confidence, 4),
                    bbox=self._plate_box(extras, plate, vehicle.frame_id, det.bbox),
                    plate_track_id=f"plate-{plate.track_id}",
                    plate_number=plate.number,
                    plate_confidence=round(plate.plate_confidence, 4) if plate.plate_confidence is not None else None,
                )

        lines = []
        if self._scheduler is not None:
            ocr_plate, lines = self._read_text(key)
            if fields.get("plate_number") is None:
                # No plate model (or it read nothing): the vehicle_text OCR plate.
                fields["plate_confirmed"] = ocr_plate is not None
                if ocr_plate is not None:
                    fields.update(plate_number=_plate_text(ocr_plate[0]), plate_confidence=round(ocr_plate[1], 4))
            fields["text_detections"] = [
                {"text_type": None, "text_value": line.text.strip(), "confidence": round(line.confidence, 3)}
                for line in lines
            ] or None

        if self._violation_classes:
            found = [
                build_violation(violation_type, round(conf, 4))
                for violation_type, (frames, conf) in (extras.violations.items() if extras is not None else ())
                if frames >= self._violation_min_frames
            ]
            fields["violations"] = found or None

        image = self._annotate(vehicle, fields, lines)
        object_key = url = width = height = None
        # Same key as every other model: <worker_key>/frames/frame_<id>.jpg.
        upload_info = self._frame_uploader.upload_frame(self._worker_key, vehicle.frame_id, image)
        if upload_info is not None:
            object_key, url = upload_info
            height, width = (int(v) for v in image.shape[:2])

        message = build_vehicle_message(
            self._job,
            device_id=self._identity.device_id,
            camera_status=self._frame_reader.get_status(),
            frame_id=vehicle.frame_id,
            occurred_at=vehicle.occurred_at,
            media={"object_key": object_key, "url": url, "width": width, "height": height},
            camera_model_assign_id=self._assign_id,
            class_id=det.class_id,
            class_name=det.class_name,
            tracking_id=f"veh-{key}",
            detection=build_vehicle_detection(**fields),
        )
        self._detection_publisher.publish_message(message)
        self._published += 1
        self._log.info(
            "vehicle published: veh-%s %s colour=%s plate=%s text=%s violations=%s",
            key, det.class_name, fields.get("colour"), fields.get("plate_number"),
            len(fields.get("text_detections") or []), [v["violation_type"] for v in fields.get("violations") or []],
        )

    def _annotate(self, vehicle: FinishedVehicle, fields: dict, lines: list) -> np.ndarray:
        """The published frame (vehicle boxes + colour, from VehicleColorEvents)
        with each vehicle's plate box/number, violations and read text drawn on.
        Several vehicles can be published on one frame: every one finished on it
        is redrawn, so re-uploading the file keeps all of them."""
        drawn = self._drawn.setdefault(vehicle.frame_id, [])
        drawn.append((fields, lines))
        self._drawn.move_to_end(vehicle.frame_id)
        while len(self._drawn) > ANNOTATED_FRAMES_KEPT:
            self._drawn.popitem(last=False)

        image = vehicle.image.copy()
        if self._overlay is None:
            # Text labels at the vehicle label's size (frame_annotator.py),
            # not scaled up with the frame height.
            self._overlay = Overlay(TEXT_LABEL_FRAME_HEIGHT)
        for f, text_lines in drawn:
            x1, y1, x2, y2 = (int(v) for v in f["vehicle_bbox"])
            labels = list(text_lines)
            if f.get("bbox"):
                px1, py1, px2, py2 = (int(v) for v in f["bbox"])
                cv2.rectangle(image, (px1, py1), (px2, py2), PLATE_COLOR, 2)
                if f.get("plate_number"):
                    _draw_label(image, f["plate_number"], px1, py1, PLATE_COLOR)
            elif f.get("plate_number"):  # OCR plate, no plate box: listed with the text
                labels.insert(0, Line([Word(f["plate_number"], "english", f.get("plate_confidence") or 0.0, (0, 0, 0, 0))]))
            # Inside the vehicle box, under its class/colour label — a vehicle at
            # the frame's bottom edge has no room below its box.
            y = max(y1, LABEL_HEIGHT) + LABEL_HEIGHT
            for violation in f.get("violations") or []:
                _draw_label(image, violation["violation_type"].upper(), x1, y, VIOLATION_COLOR)
                y += LABEL_HEIGHT
            self._overlay.draw_lines(image, x1, y - LABEL_HEIGHT + 4, labels)  # right under the last label
        return image

    def _forget(self, now: float) -> None:
        """Keeps memory flat on a live stream (same window as VehicleColorEvents)."""
        stale = [k for k, e in self._extras.items() if now - e.last_update > self._forget_after]
        for k in stale:
            del self._extras[k]
        if self._scheduler is not None:
            with self._scheduler.lock:
                self._scheduler.events.clear()  # EdgeOCR's per-job CSV log; unbounded otherwise
                for track_id, state in list(self._scheduler.tracks.items()):
                    if now - state.last_seen > self._forget_after and not state.pending:
                        del self._scheduler.tracks[track_id]

    def _log_speed(self) -> None:
        self._speed_window_frames += 1
        elapsed = time.monotonic() - self._speed_window_start
        if elapsed < SPEED_LOG_SECONDS:
            return
        self._log.info(
            "vehicle pipeline speed: capture_fps=%.1f processing_fps=%.1f tracks=%d published=%d",
            self._frame_reader.get_capture_fps(), self._speed_window_frames / elapsed,
            self._events.active, self._published,
        )
        self._speed_window_frames = 0
        self._speed_window_start = time.monotonic()

    # ------------------------------------------------------------ cleanup
    def _release_models(self) -> None:
        for name, release in (
            ("plate enricher", lambda: self._plate_enricher and self._plate_enricher.release()),
            ("plate detector", lambda: self._plate_detector and self._plate_detector.release()),
            ("OCR scheduler", lambda: self._scheduler and self._scheduler.finish(timeout=0)),
            ("OCR runners", self._release_ocr_runners),
            ("color classifier", lambda: self._classifier and self._classifier.release()),
            ("vehicle detector", lambda: self._detector and self._detector.release()),
        ):
            try:
                release()
            except Exception:
                self._log.exception("error releasing %s", name)

    def _release_ocr_runners(self) -> None:
        if self._ocr_runners:
            from slave.processes.worker.models.vehicle_text.vehicle_text_models import release_runners
            release_runners(self._ocr_runners)
            self._ocr_runners = None

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        self._release_models()
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
