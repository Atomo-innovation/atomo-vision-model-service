"""Scene-text OCR with PaddleOCR models (ONNX/NPU): DB text-line detector -> CTC readers.

Ported from EdgeOCR (fasttext_variant/edgeocr/ppocr.py), where it replaced the older
EAST -> ViT -> PARseq engine (not ported). Lines instead of words are read, each in one pass (CTC), and there is no
script classifier: every line goes through the English and the Devanagari reader and
the more confident reading is kept. PaddleOCR has no Gujarati reader, so Gujarati is
read by PARseq (CPU, "crop readers"), only on lines the CTC readers were unsure of.

Fixed input sizes (square detector input, recognizer padded to ``rec_width``) so the
same code works for NPU models later.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

from .vehicle_detector import letterbox
from .ocr_types import OCRResult, Line, Word
from .plate_parser import parse_plate
from .ocr_runtime import is_npu

DEVANAGARI = re.compile("[ऀ-ॿ]")
DEVANAGARI_LETTER = re.compile("[ऀ-ॣ॰-ॿ]")  # without danda and digits
GUJARATI_LETTER = re.compile("[ઁ-ૣ]")  # without digits
LATIN = re.compile("[A-Za-z]")
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def load_charset(path: Path) -> list[str]:
    """PaddleOCR class list: blank, the dictionary's characters, then a space."""
    chars = path.read_text(encoding="utf-8").splitlines()
    return ["[blank]", *chars, " "]


def order_points(pts: np.ndarray) -> np.ndarray:
    """4 corners -> top-left, top-right, bottom-right, bottom-left."""
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2], pts[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], np.float32)


def crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Straighten a (possibly rotated) text box; tall boxes are turned to horizontal."""
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    w, h = max(w, 1), max(h, 1)
    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (w, h),
                               borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.rot90(crop) if h >= 1.5 * w else crop


class PaddleSceneOCR:
    def __init__(self, runners: dict, charsets: dict[str, list[str]], cfg: dict,
                 crop_readers: dict | None = None) -> None:
        """runners: "det" plus one CTC reader per script ("english", "hindi").
        crop_readers: script -> callable(list of BGR line crops) -> [(text, confidence)]."""
        self.cfg = cfg
        self.det = runners["det"]
        self.det_size = cfg.get("det_size", 640)
        self.rec_height = 48
        self.rec_width = cfg.get("rec_width", 320)
        self.readers = {script: (runners[script], charsets[script]) for script in charsets}
        self.crop_readers = crop_readers or {}
        self.last_timings: dict = {}
        lid_cfg = cfg.get("language_id", {})
        self.language_id = None
        if lid_cfg.get("enabled", False):
            from .language_id import FastTextLanguageIdentifier

            self.language_id = FastTextLanguageIdentifier(
                Path(lid_cfg["model"]), min_letters=int(lid_cfg.get("min_letters", 10)),
                min_probability=float(lid_cfg.get("min_probability", 0.55)))

    # ------------------------------------------------------------------- detection
    def detect_lines(self, image: np.ndarray) -> list[np.ndarray]:
        """Text-line quads (4 x 2, crop pixels), top to bottom, left to right."""
        canvas, scale, nw, nh = letterbox(image, self.det_size, 0)
        x = ((canvas.astype(np.float32) / 255 - DET_MEAN) / DET_STD).transpose(2, 0, 1)[None]
        prob = self.det(np.ascontiguousarray(x))[0][0, 0][:nh, :nw]
        bitmap = (prob > self.cfg.get("det_thresh", 0.3)).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ih, iw = image.shape[:2]
        quads = []
        for contour in contours[:1000]:
            (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
            if min(w, h) < 3:
                continue
            # Mean probability inside the contour, like PaddleOCR's box score.
            x0, y0, bw, bh = cv2.boundingRect(contour)
            mask = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(mask, [contour - [x0, y0]], 1)
            if cv2.mean(prob[y0:y0 + bh, x0:x0 + bw], mask)[0] < self.cfg.get("det_box_thresh", 0.6):
                continue
            # DB shrinks text regions when training; grow the box back ("unclip").
            d = w * h * self.cfg.get("det_unclip", 1.5) / (2 * (w + h))
            w, h = w + 2 * d, h + 2 * d
            if min(w, h) < 5:
                continue
            quad = cv2.boxPoints(((cx, cy), (w, h), angle)) / scale
            quad[:, 0] = quad[:, 0].clip(0, iw - 1)
            quad[:, 1] = quad[:, 1].clip(0, ih - 1)
            quad = order_points(quad)
            if quad[3, 1] - quad[0, 1] >= self.cfg.get("min_word_height", 8):
                quads.append(quad)
        return sorted(quads, key=lambda q: (round(q[:, 1].mean() / 10), q[:, 0].min()))

    # --------------------------------------------------------------------- reading
    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        width = min(self.rec_width, max(1, int(np.ceil(self.rec_height * w / h))))
        resized = cv2.resize(crop, (width, self.rec_height), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((3, self.rec_height, self.rec_width), np.float32)
        out[:, :, :width] = ((resized.astype(np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)
        return out

    def recognize(self, script: str, batch: np.ndarray) -> list[tuple[str, float]]:
        runner, charset = self.readers[script]
        step = 1 if is_npu(runner) else 16  # NPU: one line per call
        probs = np.concatenate([runner(batch[i:i + step])[0] for i in range(0, len(batch), step)])
        ids, best = probs.argmax(-1), probs.max(-1)
        results = []
        for row_ids, row_probs in zip(ids, best):
            # CTC: merge repeated classes, drop blanks.
            keep = (row_ids != 0) & np.concatenate([[True], row_ids[1:] != row_ids[:-1]])
            text = "".join(charset[i] for i in row_ids[keep] if i < len(charset)).strip()
            results.append((text, float(row_probs[keep].mean()) if keep.any() else 0.0))
        return results

    def choose(self, readings: dict[str, tuple[str, float]]) -> tuple[str, str, float]:
        """A Devanagari or Gujarati reading wins only if it is real text of that script,
        sure, and more sure than the English reading (the surest of them wins).

        Text in other scripts (Gujarati) comes out of the Devanagari reader as unsure
        Devanagari mixed with Latin letters (measured: real Hindi 0.92-0.96, others 0.5-0.65).
        """
        eng = readings.get("english")
        # Single letters and digits alone are almost always misreads.
        rules = {"hindi": (DEVANAGARI_LETTER, self.cfg.get("hindi_min_confidence", 0.8)),
                 "gujarati": (GUJARATI_LETTER, self.cfg.get("gujarati_min_confidence", 0.7))}
        sure = [(reading[1], script, reading) for script, (letters, minimum) in rules.items()
                if (reading := readings.get(script)) and len(letters.findall(reading[0])) >= 2
                and not LATIN.search(reading[0]) and reading[1] >= minimum]
        if sure:
            _, script, reading = max(sure)
            if eng is None or reading[1] > eng[1]:
                return script, *reading
        if eng:
            return "english", *eng
        # No English reader (e.g. --languages hin): drop what isn't confident.
        return next(iter(readings), "hindi"), "", 0.0

    # ------------------------------------------------------------------- pipeline
    def read_batch(self, images: list[np.ndarray], keep_all: bool = False) -> list[OCRResult]:
        """OCR several vehicle crops; one result per crop. keep_all: store every reader's
        output on the result as ``.readings`` (for comparisons)."""
        timings = {}
        start = time.perf_counter()
        owners, quads, crops = [], [], []
        for index, image in enumerate(images):
            for quad in self.detect_lines(image):
                owners.append(index)
                quads.append(quad)
                crops.append(crop_quad(image, quad))
        timings["detect_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        readings: dict[str, list] = {}
        if crops:
            batch = np.stack([self._prepare(c) for c in crops])
            for script in self.readers:
                readings[script] = self.recognize(script, batch)
        timings["recognize_ms"] = (time.perf_counter() - start) * 1000

        # Crop readers (PARseq, CPU) only on lines no CTC reader was sure of and that the
        # Devanagari reader saw as Indic text: PARseq writes plausible text even on noise.
        start = time.perf_counter()
        skip = self.cfg.get("parseq_skip_confidence", 0.9)
        indic = self.cfg.get("parseq_min_devanagari", 2)
        hindi = readings.get("hindi")
        todo = [i for i in range(len(crops))
                if max((readings[s][i][1] for s in self.readers), default=0.0) < skip
                and (hindi is None or len(DEVANAGARI_LETTER.findall(hindi[i][0])) >= indic)]
        per_height = self.cfg.get("parseq_max_chars_per_height", 2.5)
        for script, reader in self.crop_readers.items():
            readings[script] = [None] * len(crops)
            if todo:
                for i, reading in zip(todo, reader([crops[i] for i in todo])):
                    # More characters than the line can hold: invented text.
                    h, w = crops[i].shape[:2]
                    if len(reading[0]) <= per_height * w / h + 2:
                        readings[script][i] = reading
        timings["parseq_ms"] = (time.perf_counter() - start) * 1000
        timings["parseq_lines"] = len(todo) if self.crop_readers else 0
        timings["lines"] = len(crops)
        self.last_timings = timings

        words_per_image: list[list[Word]] = [[] for _ in images]
        all_per_image: list[list[dict]] = [[] for _ in images]
        for i, (owner, quad) in enumerate(zip(owners, quads)):
            per_script = {script: readings[script][i] for script in readings
                          if readings[script][i] is not None}
            script, text, confidence = self.choose(per_script)
            all_per_image[owner].append(per_script)
            if text and confidence >= self.cfg.get("min_word_confidence", 0.5):
                box = (int(quad[:, 0].min()), int(quad[:, 1].min()),
                       int(quad[:, 0].max()), int(quad[:, 1].max()))
                words_per_image[owner].append(Word(text, script, confidence, box))
        results = []
        for words, all_readings in zip(words_per_image, all_per_image):
            result = OCRResult([Line([w]) for w in words])
            if keep_all:
                result.readings = all_readings
            if self.language_id is not None:
                # Plate lines are left out of language ID.
                text = " ".join(line.text for line in result.lines
                                if parse_plate(line.text, 0) is None)
                result.language, result.language_confidence = self.language_id.predict(text)
            results.append(result)
        return results
