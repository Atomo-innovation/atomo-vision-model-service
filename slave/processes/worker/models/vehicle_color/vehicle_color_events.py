"""Per-vehicle color voting on ByteTrack tracks, and when to publish each vehicle.

Each tracked vehicle (vehicle_text/vehicle_tracker.py's stable track_id) is
color-classified up to ``max_attempts`` times — on first sight, then again
whenever its box has grown by ``regrow_ratio`` (it came closer, so a better
crop) — and the reads are voted (sum of confidences per color). A vehicle is
published exactly once: when it has left the picture for ``lost_seconds``, or
after being watched ``max_wait_seconds``, with its voted color and the frame
where it was seen largest. Same publish-once idea as vehicle_text_events.py,
so a parked or slow vehicle never reaches the Master again and again — and
a vehicle that comes back under a new track_id at the same spot (parked, or
a static false detection) is suppressed too, see _repeat().
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from slave.models.detection import Detection
from slave.processes.detection.frame_annotator import draw_detections
from slave.processes.worker.models.vehicle_text.vehicle_detector import nms
from slave.processes.worker.models.vehicle_text.vehicle_tracker import ByteTracker, Track, iou_matrix
from slave.utils.time_utils import IST

# Annotated frames kept so a vehicle finishing later on an already-published
# frame is drawn onto the same image (MinIO key is per frame). ~6 MB each at 1080p.
ANNOTATED_FRAMES_KEPT = 8
# ByteTrack's track_buffer counts tracker updates at this rate (fps / 30 * buffer):
# 30 keeps a lost vehicle for track_buffer processed frames, whatever the stream fps.
TRACKER_FPS = 30.0
_BYTETRACK_KEYS = ("low_thresh", "new_track_thresh", "match_thresh", "track_buffer")


def _center(box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def build_tracker(model_config: dict, high_thresh: float) -> ByteTracker:
    """ByteTrack from a config's tracker: block (its other keys — edge_margin_px,
    agnostic_nms_iou — are read elsewhere)."""
    tracker_cfg = model_config.get("tracker", {})
    return ByteTracker(
        high_thresh=float(high_thresh),
        fps=TRACKER_FPS,
        **{k: tracker_cfg[k] for k in _BYTETRACK_KEYS if k in tracker_cfg},
    )


def agnostic_nms(boxes: np.ndarray, scores: np.ndarray, labels: List[str], iou: float):
    """Per-class NMS lets one auto-rickshaw/van come out as both "bus" and
    "truck" (two tracks, published twice): keep only the best class per box."""
    if len(boxes) < 2:
        return boxes, scores, labels
    keep = nms(boxes, scores, iou)
    return boxes[keep], scores[keep], [labels[i] for i in keep]


@dataclass
class _VehicleState:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    last_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # latest box, any score (relinking)
    velocity: Tuple[float, float] = (0.0, 0.0)          # box center px/s, from the last two sightings
    labels: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # confident sightings per class
    attempts: int = 0
    classified_area: int = 0
    votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    best_conf: Dict[str, float] = field(default_factory=dict)
    # The frame where this vehicle was seen largest — the one published.
    best_area: int = 0
    best_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    best_score: float = 0.0
    best_frame_id: int = 0
    best_wall_time: float = 0.0
    best_image: Optional[np.ndarray] = None
    emitted: bool = False


@dataclass
class _Spot:
    label: str
    box: Tuple[int, int, int, int]
    last_seen: float


@dataclass
class FinishedVehicle:
    track_id: int
    frame_id: int                 # frame where the vehicle was seen largest
    occurred_at: str              # IST ISO time of that frame
    image: np.ndarray             # that full frame, annotated
    detection: Detection


class VehicleColorEvents:
    """Shared by vehicle_color (YOLO11n + color model) and vehicle (its own
    9-class detector + the optional color enricher): both read the same
    tracker:/relink:/publish: blocks from their config/models/*.yaml.

    class_ids: class name -> class_id for the published Detection.
    min_vehicle_score: sightings below this only keep a track alive.
    classify: color of one vehicle crop, or None to publish without color.
    color_classes: only these class names are color-classified (None = all).
    """

    def __init__(
        self,
        model_config: dict,
        classify: Optional[Callable[[np.ndarray], Tuple[str, float]]],
        class_ids: Dict[str, int],
        min_vehicle_score: float,
        color_classes: Optional[Set[str]] = None,
    ) -> None:
        color_cfg = model_config.get("color", {})
        publish_cfg = model_config.get("publish", {})
        self._classify = classify
        self._color_classes = color_classes
        self._display_name = model_config.get("display_name") or model_config["model_type"]
        self._class_ids = dict(class_ids)
        self._min_vehicle_score = float(min_vehicle_score)
        self._edge_margin = int(model_config.get("tracker", {}).get("edge_margin_px", 4))
        self._confidence_threshold = float(color_cfg.get("confidence_threshold", 0.40))
        self._min_crop_pixels = int(color_cfg.get("min_crop_pixels", 40))
        self._max_attempts = max(1, int(color_cfg.get("max_attempts", 3)))
        self._regrow_ratio = float(color_cfg.get("regrow_ratio", 1.4))
        self._lost_seconds = float(publish_cfg.get("lost_seconds", 2.0))
        self._max_wait = float(publish_cfg.get("max_wait_seconds", 30))
        self._forget_after = float(publish_cfg.get("forget_after_seconds", 120))
        self._same_spot_iou = float(publish_cfg.get("same_spot_iou", 0.6))
        self._same_spot_seconds = float(publish_cfg.get("same_spot_seconds", 300))
        relink_cfg = model_config.get("relink", {})
        self._relink_gap = float(relink_cfg.get("max_gap_seconds", 1.5))
        self._relink_distance = float(relink_cfg.get("max_distance_ratio", 1.0))
        self._relink_size = float(relink_cfg.get("max_size_ratio", 2.5))
        self._states: Dict[int, _VehicleState] = {}  # keyed by the vehicle's first track_id
        self._aliases: Dict[int, int] = {}  # later track_id -> first track_id, see _relink()
        self._spots: List[_Spot] = []  # recently published vehicles, see _repeat()
        self._annotated: "OrderedDict[int, Tuple[np.ndarray, List[Detection]]]" = OrderedDict()

    @property
    def active(self) -> int:
        return sum(1 for s in self._states.values() if not s.emitted)

    def key_for(self, track_id: int) -> int:
        """The vehicle a ByteTrack track_id belongs to (its first track_id,
        see _relink()) — the track_id FinishedVehicle carries."""
        return self._aliases.get(track_id, track_id)

    def track_ids_for(self, key: int) -> List[int]:
        """Every ByteTrack track_id this vehicle has had (key first)."""
        return [key] + [alias for alias, target in self._aliases.items() if target == key]

    def observe(self, tracks: List[Track], frame: np.ndarray, frame_id: int, now: float, wall_time: float) -> None:
        """Update every visible track; classify the ones that are due a (better) color read."""
        snapshot = None  # one copy of the frame shared by every vehicle that is largest on it
        seen_now = {self._aliases.get(t.track_id, t.track_id) for t in tracks}
        for track in tracks:
            key = self._aliases.get(track.track_id, track.track_id)
            state = self._states.get(key)
            if state is None:
                state = self._relink(track, now, seen_now)
                if state is not None:
                    self._aliases[track.track_id] = state.track_id
                    seen_now.add(state.track_id)
                else:
                    state = _VehicleState(track.track_id, track.label, now, now)
                    self._states[track.track_id] = state
            dt = now - state.last_seen
            if dt > 0 and state.last_box[2] > 0:
                (px, py), (cx, cy) = _center(state.last_box), _center(track.box)
                state.velocity = ((cx - px) / dt, (cy - py) / dt)
            state.last_seen = now
            state.last_box = track.box
            if state.emitted:
                continue  # already published; just kept alive so it isn't re-published
            # ByteTrack also continues tracks on low-score boxes (down to
            # tracker.low_thresh) so IDs survive occlusions; those boxes only
            # keep the track alive — the published frame and the color reads
            # come from confident sightings only, so a track that never
            # reaches vehicle.confidence is never published. Same for boxes
            # touching the frame border (vehicle.edge_margin_px): a cut-off
            # vehicle gives odd boxes (a second track on its visible half) and
            # a bad color crop; it counts once it is fully in view.
            if track.confidence < self._min_vehicle_score or self._at_edge(track.box, frame.shape):
                continue
            # YOLO flips between close classes (bus/truck, car/truck) on one
            # vehicle: the published class is the most-seen one.
            state.labels[track.label] += 1
            state.label = max(state.labels, key=state.labels.get)

            area = track.area
            if area > state.best_area:
                if snapshot is None:
                    snapshot = frame.copy()
                state.best_area = area
                state.best_box = track.box
                state.best_score = track.confidence
                state.best_frame_id = frame_id
                state.best_wall_time = wall_time
                state.best_image = snapshot

            if self._due(state, track):
                x1, y1, x2, y2 = track.box
                h, w = frame.shape[:2]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                state.attempts += 1
                state.classified_area = area
                if crop.size:
                    name, conf = self._classify(crop)
                    if conf >= self._confidence_threshold:
                        state.votes[name] += conf
                        state.best_conf[name] = max(state.best_conf.get(name, 0.0), conf)

    def poll(self, now: float, hold: Optional[Callable[[int], bool]] = None) -> List[FinishedVehicle]:
        """Vehicles whose visit ended since the last poll, each returned once.

        hold: a vehicle (by track_id) whose result is still being worked out
        elsewhere (e.g. an OCR read in flight) waits while it returns True —
        at most one more lost_seconds, so a stuck reader never blocks it.
        """
        finished = []
        self._spots = [s for s in self._spots if now - s.last_seen <= self._same_spot_seconds]
        for track_id, state in list(self._states.items()):
            gone = now - state.last_seen
            if state.emitted:
                if gone > self._forget_after:
                    del self._states[track_id]  # keeps memory flat on a live stream
                    self._aliases = {k: v for k, v in self._aliases.items() if v != track_id}
                continue
            if gone < self._lost_seconds and now - state.first_seen < self._max_wait:
                continue
            if hold is not None and gone < 2 * self._lost_seconds and hold(track_id):
                continue
            state.emitted = True
            if state.best_image is None or self._repeat(state):
                state.best_image = None
                continue
            finished.append(self._finish(state))
            # Kept (emitted) until forget_after_seconds: ByteTrack can re-find a
            # briefly-lost track, which must not publish the vehicle again.
            state.best_image = None
        return finished

    def _at_edge(self, box: Tuple[int, int, int, int], shape) -> bool:
        m = self._edge_margin
        return box[0] <= m or box[1] <= m or box[2] >= shape[1] - m or box[3] >= shape[0] - m

    def _relink(self, track: Track, now: float, seen_now: set) -> Optional[_VehicleState]:
        """ByteTrack loses fast vehicles at low fps (the box moves too far between
        frames to overlap) and starts a new track_id a frame later. A new track
        that appears within relink.max_gap_seconds of another vehicle vanishing,
        close to where that vehicle should be by now (its last center moved on
        at its last speed — a fast vehicle missed for a few frames reappears
        well past its last box; center distance <= max_distance_ratio x that
        box's longer side, also accepted from its last center for a vehicle
        that stopped) and of similar size (area ratio <= max_size_ratio), is
        that same vehicle — it continues its state instead of being published
        twice. Not yet published vehicles only wait lost_seconds, so this has
        to happen before then: max_gap_seconds must stay below publish.lost_seconds.
        """
        best, best_dist = None, None
        nx, ny = _center(track.box)
        for key, state in self._states.items():
            gap = now - state.last_seen
            if key in seen_now or gap > self._relink_gap:
                continue
            x1, y1, x2, y2 = state.last_box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            (lx, ly), (vx, vy) = _center(state.last_box), state.velocity
            dist = min(
                ((nx - (lx + vx * gap)) ** 2 + (ny - (ly + vy * gap)) ** 2) ** 0.5,
                ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5,
            )
            size = track.area / (w * h)
            if dist > self._relink_distance * max(w, h) or not 1 / self._relink_size <= size <= self._relink_size:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = state, dist
        return best

    def _repeat(self, state: _VehicleState) -> bool:
        """A parked vehicle (or a static false detection) that drops out for
        longer than ByteTrack's buffer comes back under a new track_id. If the
        same class was already published at (almost) the same box within
        same_spot_seconds, it is the same thing: not published again, and the
        spot's clock restarts so it stays suppressed while it keeps standing there.
        """
        box = np.array([state.best_box], dtype=np.float32)
        for spot in self._spots:
            if spot.label == state.label and \
                    iou_matrix(box, np.array([spot.box], dtype=np.float32))[0, 0] >= self._same_spot_iou:
                spot.last_seen, spot.box = state.last_seen, state.best_box
                return True
        self._spots.append(_Spot(state.label, state.best_box, state.last_seen))
        return False

    def _due(self, state: _VehicleState, track: Track) -> bool:
        if self._classify is None or (self._color_classes is not None and track.label not in self._color_classes):
            return False
        x1, y1, x2, y2 = track.box
        if min(x2 - x1, y2 - y1) < self._min_crop_pixels or state.attempts >= self._max_attempts:
            return False
        return state.attempts == 0 or track.area >= state.classified_area * self._regrow_ratio

    def _finish(self, state: _VehicleState) -> FinishedVehicle:
        color = color_conf = None
        if state.votes:
            color = max(state.votes, key=state.votes.get)
            color_conf = round(state.best_conf[color], 4)
        detection = Detection(
            class_id=self._class_ids.get(state.label, -1),
            class_name=state.label,
            confidence=float(state.best_score),
            bbox=tuple(float(v) for v in state.best_box),
            model_name=self._display_name,
            extra={"track_id": state.track_id, "color": color, "color_confidence": color_conf},
        )
        # Several vehicles can be largest on the same frame: all are drawn onto
        # one image, so re-uploading it keeps every earlier vehicle's box on the
        # file their messages point to.
        raw, drawn = self._annotated.get(state.best_frame_id, (state.best_image, []))
        drawn = drawn + [detection]
        self._annotated[state.best_frame_id] = (raw, drawn)
        self._annotated.move_to_end(state.best_frame_id)
        while len(self._annotated) > ANNOTATED_FRAMES_KEPT:
            self._annotated.popitem(last=False)
        return FinishedVehicle(
            track_id=state.track_id,
            frame_id=state.best_frame_id,
            occurred_at=datetime.fromtimestamp(state.best_wall_time, IST).isoformat(),
            image=draw_detections(raw, drawn),
            detection=detection,
        )
