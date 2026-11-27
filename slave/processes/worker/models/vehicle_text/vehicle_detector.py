"""YOLO vehicle detection from the exported ONNX / NPU model (no Ultralytics needed)."""

from __future__ import annotations

import cv2
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns kept indices, best score first."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def letterbox(image: np.ndarray, size: int, pad_value: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio into a size x size canvas (image at top-left)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=interpolation)
    return canvas, scale, nw, nh


class VehicleDetector:
    def __init__(self, runner, imgsz: int, classes: dict[int, str], min_score: float,
                 nms_iou: float = 0.7) -> None:
        self.runner = runner
        self.imgsz = imgsz
        self.class_ids = np.array(sorted(classes))
        self.labels = [classes[c] for c in self.class_ids]
        self.min_score = min_score
        self.nms_iou = nms_iou

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (boxes N x 4 xyxy in frame pixels, scores N, labels N)."""
        canvas, scale, _, _ = letterbox(frame, self.imgsz, 114)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255
        # Output (1, 4 + 80, anchors): cx, cy, w, h, then one sigmoid score per COCO class.
        pred = self.runner(x)[0][0].T
        scores_all = pred[:, 4 + self.class_ids]
        best = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(best)), best]
        mask = scores >= self.min_score
        if not mask.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        cx, cy, w, h = pred[mask, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        scores, best = scores[mask], best[mask]
        # Offset boxes by class so NMS never suppresses across classes.
        keep = nms(boxes + best[:, None] * 4096.0, scores, self.nms_iou)
        boxes = boxes[keep] / scale
        fh, fw = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh)
        return boxes.astype(np.float32), scores[keep], [self.labels[i] for i in best[keep]]
