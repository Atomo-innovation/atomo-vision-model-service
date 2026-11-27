"""Draws vehicle boxes and their OCR text (Gujarati/Hindi need real fonts, not cv2.putText)."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ocr_types import Line
from .ocr_scheduler import OCRScheduler
from .vehicle_tracker import Track

FONTS = {
    "gujarati": "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
    "hindi": "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "english": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

READ_COLOR = (60, 210, 60)      # BGR: vehicle has text
PENDING_COLOR = (0, 190, 255)   # BGR: OCR in flight / nothing read yet


@lru_cache(maxsize=16)
def _font(script: str, size: int) -> ImageFont.FreeTypeFont:
    # Boards often lack the Noto/DejaVu fonts; Pillow's built-in font then keeps the
    # overlay working (Latin only - install fonts-noto-core for Gujarati/Hindi).
    for path in (FONTS.get(script, FALLBACK_FONT), FALLBACK_FONT):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


@lru_cache(maxsize=512)
def _render_label(words: tuple[tuple[str, str], ...], size: int) -> np.ndarray:
    """Render one text line (each word in its own script's font) as a BGRA image."""
    space = size // 3
    pad = max(3, size // 5)
    widths, top, bottom = [], 0, 0
    for text, script in words:
        left, t, right, b = _font(script, size).getbbox(text)
        widths.append(right - left)
        top, bottom = min(top, t), max(bottom, b)
    width = sum(widths) + space * (len(words) - 1) + 2 * pad
    height = bottom - top + 2 * pad
    image = Image.new("RGBA", (width, height), (0, 0, 0, 180))
    draw = ImageDraw.Draw(image)
    x = pad
    for (text, script), w in zip(words, widths):
        font = _font(script, size)
        draw.text((x - font.getbbox(text)[0], pad - top), text, font=font, fill=(255, 235, 0, 255))
        x += w + space
    rgba = np.asarray(image)
    return np.dstack([rgba[..., 2::-1], rgba[..., 3]])  # RGBA -> BGRA


def _blend(frame: np.ndarray, label: np.ndarray, x: int, y: int) -> None:
    fh, fw = frame.shape[:2]
    lh, lw = label.shape[:2]
    x, y = max(0, min(x, fw - lw)), max(0, min(y, fh - lh))
    x2, y2 = min(fw, x + lw), min(fh, y + lh)
    if x2 <= x or y2 <= y:
        return
    patch = label[: y2 - y, : x2 - x]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    roi = frame[y:y2, x:x2]
    roi[:] = (patch[..., :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


class Overlay:
    def __init__(self, frame_height: int, max_lines: int = 4) -> None:
        self.font_size = max(16, frame_height // 36)
        self.thickness = max(2, frame_height // 400)
        self.max_lines = max_lines

    def draw(self, frame: np.ndarray, tracks: list[Track], scheduler: OCRScheduler,
             hud: str | None = None) -> np.ndarray:
        for track in tracks:
            lines, pending = scheduler.display_lines(track.track_id)
            self.draw_vehicle(frame, track.box, f"{track.label} #{track.track_id}", lines, pending)
        if hud:
            self.draw_hud(frame, hud)
        return frame

    def draw_vehicle(self, frame: np.ndarray, box, tag: str, lines: list[Line],
                     pending: bool = False) -> None:
        color = READ_COLOR if lines else PENDING_COLOR
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        tag += " reading..." if pending and not lines else ""
        cv2.putText(frame, tag, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size / 32, color, self.thickness, cv2.LINE_AA)
        y = y2 + 4
        for line in lines[: self.max_lines]:
            label = self._label(line)
            _blend(frame, label, x1, y)
            y += label.shape[0] + 2

    def draw_hud(self, frame: np.ndarray, hud: str) -> None:
        cv2.putText(frame, hud, (10, 10 + self.font_size), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size / 32, (255, 255, 255), self.thickness, cv2.LINE_AA)

    def _label(self, line: Line) -> np.ndarray:
        return _render_label(tuple((w.text, w.script) for w in line.words), self.font_size)
