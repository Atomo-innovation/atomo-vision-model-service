"""Standalone Vehicle Text `.onnx` worker -- a jobs.yaml entry with
model_type: "vehicle_text", entirely independent of any other job.
Same frame-loop shape as number_plate_pt_worker.py; replaces EdgeOCR's
app.py main loop:

    frame -> YOLO vehicles -> ByteTrack -> OCRScheduler (background OCR
    thread: PP-OCR lines -> English/Hindi/Gujarati readers -> voting)
          -> VehicleTextEvents: once a vehicle's reading is final, one
             annotated MinIO frame + one NATS detection for it

Unlike the frame-by-frame models, nothing is published for frames without
a finished vehicle. All tracker/scheduler state is per-job, never shared
(CLAUDE.md SS20).
"""

from __future__ import annotations

import os
import threading
import time

from slave.logging.logger import get_logger
from slave.models.detection import group_detections
from slave.models.job import JobConfig, ModelAssignment
from slave.processes.detection.detection_result import DetectionResult
from slave.processes.minio.frame_uploader import FrameUploader
from slave.processes.publisher.detection_publisher import DetectionPublisher
from slave.processes.stream.frame_reader import FrameReader
from slave.processes.video.video_manager import VideoManager
from slave.processes.worker.models.vehicle_text.color_scheduler import ColorScheduler
from slave.processes.worker.models.vehicle_text.ocr_scheduler import OCRScheduler
from slave.processes.worker.models.vehicle_text.onnx.vehicle_text_onnx_detector import VehicleTextOnnxDetector
from slave.processes.worker.models.vehicle_text.vehicle_text_events import FinishedVehicle, VehicleTextEvents
from slave.processes.worker.models.vehicle_text.vehicle_tracker import ByteTracker
from slave.processes.worker.worker_factory import WorkerIdentity

FRAME_POLL_TIMEOUT_SECONDS = 1.0
SPEED_LOG_SECONDS = 10.0
# ByteTrack's track_buffer counts tracker updates at this rate (fps / 30 * buffer):
# 30 keeps a lost vehicle for track_buffer processed frames, whatever the stream fps.
TRACKER_FPS = 30.0


class VehicleTextOnnxWorker:
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

        # Weights live in this job type's own folder (see vehicle_text/README.md).
        model_path = os.path.join(models_dir, "onnx", assignment.model_name)
        self._detector = VehicleTextOnnxDetector(model_path, model_config, assignment.model_name)
        self._tracker = ByteTracker(
            high_thresh=float(model_config["vehicle"]["confidence"]),
            fps=TRACKER_FPS,
            **model_config["tracker"],
        )
        # keep_frames: the publisher needs the full frame of each vehicle's best read.
        self._scheduler = OCRScheduler(
            self._detector.ocr, model_config["scheduler"], model_config["ocr"], keep_frames=True
        )
        config_path = os.path.abspath(
            os.path.join(models_dir, "..", "..", "..", "..", "config", "models", "vehicle_text.yaml")
        )
        self._color_scheduler = ColorScheduler(
            models_dir,
            model_config.get("color", {}),
            model_extension=assignment.model_extension,
            config_path=config_path,
        )
        self._events = VehicleTextEvents(
            model_config.get("publish", {}), model_config.get("display_name", assignment.model_name)
        )

        self._frame_reader = FrameReader(worker_key, job.camera_stream, decode_every_n_frames=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"vehicle-text-onnx-{worker_key}", daemon=True
        )
        self._clock_start = time.monotonic()
        self._frames = 0
        self._published = 0
        self._speed_window_start = time.monotonic()
        self._speed_window_frames = 0

    def start(self) -> None:
        self._frame_reader.start()
        self._scheduler.start()
        self._color_scheduler.start()
        self._thread.start()
        self._log.info("vehicle_text .onnx worker started (color=%s)", self._color_scheduler.enabled)

    def set_color_enabled(self, enabled: bool) -> None:
        self._color_scheduler.set_enabled(enabled)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=10)
        self._cleanup()
        self._log.info("vehicle_text .onnx worker stopped (%d vehicle(s) published)", self._published)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        tracks = []
        while not self._stop_event.is_set():
            self._color_scheduler.maybe_reload_config()
            frame = self._frame_reader.get_latest_frame(timeout=FRAME_POLL_TIMEOUT_SECONDS)
            now = time.monotonic() - self._clock_start
            if frame is None:
                self._publish_finished(now)  # vehicles still finish while the stream is down
                continue
            try:
                if self._frames % self._detector.detect_every == 0:
                    tracks = self._tracker.update(*self._detector.detect(frame.image))
                    self._scheduler.observe(tracks, frame.image, frame.frame_id, now, frame.timestamp)
                    self._color_scheduler.observe(tracks, frame.image)
                self._frames += 1
                self._publish_finished(now)
                self._log_speed()
            except Exception:
                self._log.exception("frame processing failed for frame_id=%s", frame.frame_id)

    def _publish_finished(self, now: float) -> None:
        for vehicle in self._events.poll(self._scheduler, now, self._color_scheduler):
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
            telemetry=group_detections([vehicle.detection], self._job, tracking_id=object_key),
            object_key=object_key,
            url=url,
            width=width,
            height=height,
        )
        self._detection_publisher.publish_detection(result)
        self._published += 1
        extra = vehicle.detection.extra
        self._log.info(
            "vehicle published: track_id=%s %s color=%s plate=%s text=%r",
            vehicle.track_id, extra["vehicle_class"], extra.get("color"), extra["plate_number"], extra["text"],
        )

    def _log_speed(self) -> None:
        self._speed_window_frames += 1
        elapsed = time.monotonic() - self._speed_window_start
        if elapsed < SPEED_LOG_SECONDS:
            return
        latencies = list(self._scheduler.latencies_ms)[-20:]
        self._log.info(
            "ONNX speed: capture_fps=%.1f processing_fps=%.1f ocr_ms=%.0f ocr_queue=%d "
            "ocr_dropped=%d tracks=%d published=%d",
            self._frame_reader.get_capture_fps(),
            self._speed_window_frames / elapsed,
            sum(latencies) / len(latencies) if latencies else 0.0,
            self._scheduler.jobs.qsize(),
            self._scheduler.dropped_jobs,
            len(self._scheduler.tracks),
            self._published,
        )
        self._speed_window_frames = 0
        self._speed_window_start = time.monotonic()

    def _cleanup(self) -> None:
        try:
            self._frame_reader.stop()
        except Exception:
            self._log.exception("error stopping frame reader")
        try:
            self._color_scheduler.stop()
        except Exception:
            self._log.exception("error stopping color scheduler")
        try:
            self._scheduler.finish(timeout=0)
        except Exception:
            self._log.exception("error stopping OCR scheduler")
        try:
            self._detector.release()
        except Exception:
            self._log.exception("error releasing detector")
        try:
            self._video_manager.close()
        except Exception:
            self._log.exception("error closing video manager")
