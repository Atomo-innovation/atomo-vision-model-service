"""OCR result types shared by the OCR engine, scheduler, plates and overlay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LANG_TO_SCRIPT = {"eng": "english", "hin": "hindi", "guj": "gujarati"}
# ViT classes whose script is read by another recognizer (Marathi is Devanagari).
SCRIPT_ALIASES = {"marathi": "hindi"}


@dataclass
class Word:
    text: str
    script: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in crop coordinates


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def script(self) -> str:
        counts: dict[str, int] = {}
        for word in self.words:
            counts[word.script] = counts.get(word.script, 0) + len(word.text)
        return max(counts, key=counts.get)

    @property
    def confidence(self) -> float:
        return sum(word.confidence for word in self.words) / len(self.words)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (min(w.box[0] for w in self.words), min(w.box[1] for w in self.words),
                max(w.box[2] for w in self.words), max(w.box[3] for w in self.words))


@dataclass
class OCRResult:
    lines: list[Line] = field(default_factory=list)
    language: str = "und"
    language_confidence: float = 0.0

    @property
    def confidence(self) -> float:
        words = [w for line in self.lines for w in line.words]
        return sum(w.confidence for w in words) / len(words) if words else 0.0

    @property
    def quality(self) -> float:
        # Prefer confident reads, then longer ones (a vehicle seen closer shows more text).
        chars = sum(len(line.text) for line in self.lines)
        return self.confidence * float(np.log1p(chars))


def group_lines(words: list[Word]) -> list[Line]:
    """Group word boxes into reading-order lines (top to bottom, left to right)."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.box[1] + w.box[3]) / 2):
        cy = (word.box[1] + word.box[3]) / 2
        height = word.box[3] - word.box[1]
        for line in lines:
            line_top = min(w.box[1] for w in line)
            line_bottom = max(w.box[3] for w in line)
            line_cy = (line_top + line_bottom) / 2
            if abs(cy - line_cy) <= 0.5 * max(height, line_bottom - line_top):
                line.append(word)
                break
        else:
            lines.append([word])
    return [Line(sorted(line, key=lambda w: w.box[0])) for line in lines]
