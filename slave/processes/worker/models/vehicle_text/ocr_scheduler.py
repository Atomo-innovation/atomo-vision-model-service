"""Decides when each tracked vehicle needs OCR and runs OCR on a background thread.

The video loop never waits for OCR: it hands vehicle crops to a queue and keeps drawing
the latest known text for every track. Each vehicle is read when it first appears, again
when it has grown noticeably (came closer), and periodically until its number plate has
been read the same way ``plate_confirmations`` times and some other text (stickers,
names, slogans) has been read too, capped at ``max_attempts``. Plate
reads from all frames are combined by ``PlateVoter``.
"""

from __future__ import annotations

import math
import re
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from .plate_parser import PlateVoter, parse_plate, plate_from_lines
from .ocr_types import Line, OCRResult, Word
from .vehicle_tracker import Track


@dataclass
class TrackState:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    first_frame: int
    last_frame: int
    attempts: int = 0
    pending: bool = False
    last_area: int = 0
    last_submit: float = -math.inf
    best: OCRResult | None = None
    best_frame: int | None = None
    best_crop: np.ndarray | None = None
    plates: PlateVoter = field(default_factory=PlateVoter)
    # Only filled when events are published (keep_frames): the full frame of the best read.
    best_image: np.ndarray | None = None
    best_origin: tuple[int, int] = (0, 0)   # crop's top-left corner in the frame
    best_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    best_video_time: float = 0.0
    best_wall_time: float = 0.0
    emitted: bool = False                   # an event was published for this vehicle
    event_id: str | None = None             # its id/keys; re-publishing overwrites them
    event_base: str | None = None
    published: tuple | None = None          # (frame, plate, text) last published
    # Non-plate text seen across reads: [normalized text, how many reads had it].
    text_votes: list = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


TEXT_MATCH = 0.8  # "perfect" and "porfect" are the same text read twice


@dataclass
class Job:
    track_id: int
    crop: np.ndarray
    frame_index: int
    video_time: float
    wall_time: float
    box: tuple[int, int, int, int]
    origin: tuple[int, int]
    frame: np.ndarray | None = None  # full frame, only with keep_frames
    submitted: float = field(default_factory=time.perf_counter)


class OCRScheduler:
    def __init__(self, ocr, sched_cfg: dict, ocr_cfg: dict,
                 block_when_full: bool = False, wait_for_ocr: bool = False,
                 keep_frames: bool = False) -> None:
        self.ocr = ocr
        self.cfg = sched_cfg
        self.crop_padding = ocr_cfg["crop_padding"]
        self.display_confidence = ocr_cfg["display_confidence"]
        self.min_display_chars = ocr_cfg["min_display_chars"]
        self.max_jobs_per_batch = ocr_cfg["max_jobs_per_batch"]
        self.jobs: queue.Queue[Job] = queue.Queue(maxsize=sched_cfg["queue_size"])
        self.tracks: dict[int, TrackState] = {}
        self.events: list[dict] = []
        self.latencies_ms: deque[float] = deque(maxlen=200)
        # Files: wait for room in the queue so no vehicle is skipped. Live: never stall
        # the camera; count the jobs that had to be dropped instead.
        self.block_when_full = block_when_full
        # Offline (--fast) runs decode far faster than real time; without waiting, a
        # vehicle can cross the whole frame before its first OCR result returns.
        self.wait_for_ocr = wait_for_ocr
        # Event publishing needs the whole frame of each vehicle's best read.
        self.keep_frames = keep_frames
        self.dropped_jobs = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ocr-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    # ------------------------------------------------------------ video thread side
    def observe(self, tracks: list[Track], frame: np.ndarray, frame_index: int,
                video_time: float, wall_time: float | None = None) -> None:
        height, width = frame.shape[:2]
        wall_time = time.time() if wall_time is None else wall_time
        snapshot = None  # one copy of the frame shared by this frame's jobs (keep_frames)
        for track in tracks:
            with self.lock:
                state = self.tracks.get(track.track_id)
                if state is None:
                    state = TrackState(track.track_id, track.label, video_time, video_time,
                                       frame_index, frame_index)
                    self.tracks[track.track_id] = state
                state.last_seen, state.last_frame, state.label = video_time, frame_index, track.label
                if not self._should_submit(state, track, video_time):
                    continue
                state.pending = True

            x1, y1, x2, y2 = track.box
            pad_x = int((x2 - x1) * self.crop_padding)
            pad_y = int((y2 - y1) * self.crop_padding)
            cx, cy = max(0, x1 - pad_x), max(0, y1 - pad_y)
            region = (slice(cy, min(height, y2 + pad_y)), slice(cx, min(width, x2 + pad_x)))
            if self.keep_frames:
                if snapshot is None:
                    snapshot = frame.copy()  # the caller draws on frame afterwards
                crop = snapshot[region]
            else:
                crop = frame[region].copy()
            job = Job(track.track_id, crop, frame_index, video_time, wall_time, track.box,
                      (cx, cy), snapshot)
            try:
                self.jobs.put(job, block=self.block_when_full)
            except queue.Full:
                # OCR is saturated; try this vehicle again on a later frame.
                with self.lock:
                    state.pending = False
                    self.dropped_jobs += 1
                continue
            with self.lock:
                state.attempts += 1
                state.last_area = track.area
                state.last_submit = video_time
        if self.wait_for_ocr:
            self.jobs.join()

    def _should_submit(self, state: TrackState, track: Track, now: float) -> bool:
        if state.pending or state.attempts >= self.cfg["max_attempts"]:
            return False
        if track.box[3] - track.box[1] < self.cfg["min_vehicle_height"]:
            return False
        if state.attempts == 0:
            return True
        if self.reading_done(state):
            return False
        if track.area >= state.last_area * self.cfg["regrow_ratio"]:
            return True
        return now - state.last_submit >= self.cfg["retry_seconds"]

    def plate(self, state: TrackState) -> tuple[str, float, int] | None:
        """(plate, confidence, agreeing reads) for a track, formatted for display."""
        best = state.plates.best()
        if best is None:
            return None
        return parse_plate(best[0], best[1]).pretty, best[1], best[2]

    def reading_done(self, state: TrackState) -> bool:
        """Plate confirmed by several frames and other text read too (caller holds the lock).

        Stopping at the plate alone would skip the rest of the vehicle's writing whenever
        OCR is slow (board CPU), since the plate is usually confirmed first.
        """
        plate = state.plates.best()
        if plate is None or plate[2] < self.cfg["plate_confirmations"]:
            return False
        need = self.cfg["plate_confirmations"]
        return any(parse_plate(line.text, 0) is None and self.text_reads(state, line) >= need
                   for line in self.text_lines(state))

    def text_lines(self, state: TrackState) -> list[Line]:
        """Confident lines of a track's best read (caller holds the lock)."""
        return [] if state.best is None else self._confident(state.best.lines)

    def _confident(self, lines: list[Line]) -> list[Line]:
        return [line for line in lines
                if line.confidence >= self.display_confidence
                and len(line.text.replace(" ", "")) >= self.min_display_chars]

    def text_reads(self, state: TrackState, line: Line) -> int:
        """In how many reads this (non-plate) line appeared, allowing small misreads."""
        key = _norm(line.text)
        return max((count for text, count in state.text_votes
                    if SequenceMatcher(None, text, key).ratio() >= TEXT_MATCH), default=0)

    def _vote_text(self, state: TrackState, result: OCRResult) -> None:
        seen = set()
        for line in self._confident(result.lines):
            key = _norm(line.text)
            if not key or key in seen or parse_plate(line.text, 0) is not None:
                continue
            seen.add(key)
            for vote in state.text_votes:
                if SequenceMatcher(None, vote[0], key).ratio() >= TEXT_MATCH:
                    vote[1] += 1
                    break
            else:
                state.text_votes.append([key, 1])

    def display_lines(self, track_id: int) -> tuple[list[Line], bool]:
        """Lines to draw for a track, and whether an OCR job for it is in flight."""
        with self.lock:
            state = self.tracks.get(track_id)
            if state is None:
                return [], False
            return self.state_display_lines(state), state.pending

    def state_display_lines(self, state: TrackState) -> list[Line]:
        """Voted plate first, then the other confident lines (caller holds the lock)."""
        # Raw plate reads are dropped; the voted plate goes first instead.
        lines = [line for line in self.text_lines(state) if parse_plate(line.text, 0) is None]
        plate = self.plate(state)
        if plate is not None:
            lines.insert(0, Line([Word(plate[0], "english", plate[1], (0, 0, 0, 0))]))
        return lines

    # ----------------------------------------------------------- OCR thread side
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                batch = [self.jobs.get(timeout=0.1)]
            except queue.Empty:
                continue
            while len(batch) < self.max_jobs_per_batch:
                try:
                    batch.append(self.jobs.get_nowait())
                except queue.Empty:
                    break
            try:
                results: list[OCRResult | None] = self.ocr.read_batch([job.crop for job in batch])
            except Exception:
                traceback.print_exc()
                results = [None] * len(batch)
            done = time.perf_counter()
            with self.lock:
                for job, result in zip(batch, results):
                    state = self.tracks[job.track_id]
                    state.pending = False
                    latency = (done - job.submitted) * 1000
                    self.latencies_ms.append(latency)
                    plate = plate_from_lines(result.lines) if result else None
                    if plate is not None:
                        state.plates.add(plate)
                    if result:
                        self._vote_text(state, result)
                    self.events.append({
                        "track_id": job.track_id,
                        "frame_index": job.frame_index,
                        "video_time_s": round(job.video_time, 3),
                        "latency_ms": round(latency, 1),
                        "text": " | ".join(line.text for line in result.lines) if result else "",
                        "confidence": round(result.confidence, 3) if result else 0.0,
                        "plate": plate.text if plate else "",
                        "error": result is None,
                    })
                    if result and result.lines and (
                        state.best is None or result.quality > state.best.quality
                    ):
                        state.best, state.best_frame, state.best_crop = result, job.frame_index, job.crop
                        if job.frame is not None:
                            state.best_image, state.best_origin = job.frame, job.origin
                            state.best_box = job.box
                            state.best_video_time, state.best_wall_time = job.video_time, job.wall_time
            for _ in batch:
                self.jobs.task_done()

    def finish(self, timeout: float) -> None:
        """Wait (up to ``timeout`` seconds) for queued OCR jobs, then stop the worker."""
        deadline = time.monotonic() + timeout
        while self.jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=5)
