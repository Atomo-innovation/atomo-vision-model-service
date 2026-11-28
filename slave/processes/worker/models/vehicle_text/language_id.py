"""Lightweight FastText language identification for OCR output text."""

from __future__ import annotations

import re
from pathlib import Path


class FastTextLanguageIdentifier:
    """Identify English, Hindi, or Gujarati after OCR has produced enough text."""

    ALLOWED = {"en", "hi", "gu"}

    def __init__(self, model_path: Path, min_letters: int = 10,
                 min_probability: float = 0.55) -> None:
        try:
            import fasttext
        except ImportError as exc:
            raise RuntimeError(
                "FastText language ID is enabled, but fasttext is not installed. "
                "Install this variant's requirements.txt."
            ) from exc
        if not model_path.is_file():
            raise FileNotFoundError(
                f"FastText model not found: {model_path}. See processes/worker/models/vehicle_text/README.md."
            )
        self.model = fasttext.load_model(str(model_path))
        self.min_letters = min_letters
        self.min_probability = min_probability

    def predict(self, text: str) -> tuple[str, float]:
        """Return ISO language code and probability; use ``und`` when uncertain."""
        cleaned = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        cleaned = " ".join(cleaned.split())
        letters = sum(char.isalpha() for char in cleaned)
        if letters < self.min_letters:
            return "und", 0.0

        labels, probabilities = self.model.predict(cleaned.replace("\n", " "), k=1)
        if not labels:
            return "und", 0.0
        language = labels[0].removeprefix("__label__")
        probability = float(probabilities[0])
        if language not in self.ALLOWED or probability < self.min_probability:
            return "und", probability
        return language, probability
