"""Standalone Vehicle Color `.nb` worker — a jobs.yaml entry with
model_type: "vehicle_color", entirely independent of any other job.
Same frame-loop shape as vehicle_text_nb_worker.py, minus the OCR:

    frame -> YOLO11n vehicles (NPU) -> ByteTrack -> a few color reads (NPU) per
             track_id, voted (VehicleColorEvents)
          -> once a vehicle has left (or was watched max_wait_seconds),
             one annotated MinIO frame + one NATS detection for it

Nothing is published for frames without a finished vehicle, so a vehicle
standing in view is never reported again and again. All tracker/vote
state is per-job, never shared (CLAUDE.md §20). No face_controller here —
worker_factory.py only attaches one to "person" and "safety" model types.
"""

from __future__ import annotations

import threading
import time

from slave.logging.logger import get_logger
from slave.models.detection import Detection, group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.vehicle_color.nb.vehicle_color_nb_detector import VehicleColorNbDetector
from slave.processes.worker.models.vehicle_color.vehicle_color_events import (
    FinishedVehicle,
    VehicleColorEvents,
    build_tracker,
)
from slave.processes.worker.worker_factory import WorkerIdentity

FRAME_POLL_TIMEOUT_SECONDS = 1.0


class VehicleColorNbWorker:
    def __init__(
        self,
        job: JobConfig,
        assignment: ModelAssignment,
        worker_key: str,
        model_config: dict,
        models_dir: str,
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

        # The job's model_name only selects this job type; the model files
        # are named in config/models/vehicle_color.yaml's nb: block.
        self._detector = VehicleColorNbDetector(models_dir, model_config, assignment.model_name)
        min_score = float(model_config["vehicle"]["confidence"])
        self._tracker = build_tracker(model_config, min_score)
        self._events = VehicleColorEvents(
            model_config,
            self._detector.classify,
            class_ids={str(v): int(k) for k, v in model_config["vehicle"]["classes"].items()},
            min_vehicle_score=min_score,
        )

        url = job.camera_stream
        self._frame_reader = FrameReader(worker_key, url)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"vehicle-color-nb-{worker_key}", daemon=True)
        self._clock_start = time.monotonic()
        self._published = 0

    def start(self) -> None:
        self._frame_reader.start()
        self._thread.start()
        self._log.info("vehicle_color .nb worker started")

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("vehicle_color .nb worker stopped (%d vehicle(s) published)", self._published)

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
                self._process_frame(frame, now)
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _process_frame(self, frame, now: float) -> None:
        tracks = self._tracker.update(*self._detector.detect(frame.image))
        self._events.observe(tracks, frame.image, frame.frame_id, now, frame.timestamp)
        self._publish_finished(now)

        # Video only records while a vehicle is in frame, with its live track boxes.
        if tracks:
            live = [
                Detection(class_id=-1, class_name=f"{t.label} #{t.track_id}", confidence=t.confidence,
                          bbox=tuple(float(v) for v in t.box), model_name="")
                for t in tracks
            ]
            self._video_manager.feed_frame(draw_detections(frame.image, live))

    def _publish_finished(self, now: float) -> None:
        for vehicle in self._events.poll(now):
            try:
                self._publish(vehicle)
            except Exception:
                self._log.exception("failed to publish vehicle track_id=%s", vehicle.track_id)

    def _publish(self, vehicle: FinishedVehicle) -> None:
        object_key = url = None
        # Same key as every other model: <worker_key>/frames/frame_<id>.jpg.
        upload_info = self._frame_uploader.upload_frame(self._worker_key, vehicle.frame_id, vehicle.image)
        width = height = None
        if upload_info is not None:
            object_key, url = upload_info
            height, width = (int(v) for v in vehicle.image.shape[:2])

        # class_name is the vehicle class (car/bus/...), not the model_type,
        # so the assignment ids have to be looked up by model_type here.
        telemetry = group_detections(
            [vehicle.detection],
            self._job,
            tracking_id=object_key,
            assign_id=self._job.camera_model_assign_id_for("vehicle_color"),
            model_id=self._job.model_id_for("vehicle_color"),
            model_name=self._job.model_name_for("vehicle_color"),
        )

        result = DetectionResult(
            job_id=self._job.job_id,
            frame_id=vehicle.frame_id,
            occurred_at=vehicle.occurred_at,
            device_name=self._identity.device_name,
            branch_id=self._job.branch_id,
            branch_name=self._job.branch_name,
            device_id=self._identity.device_id,
            zone_id=self._job.zone_id,
            zone_name=self._job.zone_name,
            camera_id=self._job.camera_id,
            camera_name=self._job.camera_name,
            camera_stream=self._job.camera_stream,
            camera_status=self._frame_reader.get_status(),
            latitude=self._job.latitude,
            longitude=self._job.longitude,
            telemetry=telemetry,
            object_key=object_key,
            url=url,
            width=width,
            height=height,
        )
        self._detection_publisher.publish_detection(result)
        self._published += 1
        extra = vehicle.detection.extra
        self._log.info(
            "vehicle published: track_id=%s %s color=%s (%s)",
            vehicle.track_id, vehicle.detection.class_name, extra["color"], extra["color_confidence"],
        )

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        try:
            self._detector.release()
        except Exception:
            self._log.exception("error releasing detector")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
