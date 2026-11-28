"""Indian number-plate parsing, OCR-confusion correction and per-vehicle voting.

Scene OCR reads a plate as free text, so it often swaps look-alike characters
(``gjo1rh9652``, ``cj05ra9459``, ``gj3zaa9115``). Plates follow a fixed layout:

    SS  NN  [L{0,3}]  NNNN        e.g. GJ 01 RH 9652, DL 3C AB 1234
    YY  BH  NNNN  L{1,2}          Bharat series, e.g. 22 BH 1234 AA

so every position is known to be a letter or a digit, and the state code must be a
real one. ``parse_plate`` fixes characters to fit that layout; ``PlateVoter`` combines
the reads of one vehicle over several frames instead of trusting a single frame.
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

# Character the OCR printed -> what it most likely is when a digit / letter is required.
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "T": "1", "J": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
# Extra look-alikes tried only to reach a valid state code (CJ -> GJ, G1 -> GJ, ...).
STATE_ALTERNATIVES = {"C": "G", "G": "C", "0": "OD", "O": "D", "D": "O", "1": "IJ", "I": "J",
                      "J": "I", "6": "G", "8": "B", "5": "S", "2": "Z", "4": "A", "U": "V",
                      "3": "J",  # this camera's J looks like 3: "6301HP..." = "GJ01HP..."
                      "V": "U", "K": "X", "N": "M", "M": "N", "H": "N"}

# (rto digits, series letters) -> prior cost; rarer layouts must earn their place.
# Single-digit RTO codes exist only in Delhi (DL 3C AB 1234).
LAYOUTS = {(2, 2): 0.0, (2, 1): 0.0, (2, 3): 0.5, (2, 0): 1.5,
           (1, 2): 0.0, (1, 1): 0.0, (1, 3): 0.5}
TRIM_COST = 1.0


@dataclass
class Plate:
    text: str      # canonical, e.g. "GJ01RH9652"
    cost: float    # number of corrected / trimmed characters
    confidence: float

    @property
    def score(self) -> float:
        return self.confidence / (1.0 + 0.5 * self.cost)

    @property
    def pretty(self) -> str:
        m = re.fullmatch(r"([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})", self.text)
        if m:
            return " ".join(part for part in m.groups() if part)
        m = re.fullmatch(r"(\d{2})(BH)(\d{4})([A-Z]{1,2})", self.text)
        return " ".join(m.groups()) if m else self.text


def _fit(chars: str, kinds: str) -> tuple[str, int] | None:
    """Coerce ``chars`` to a pattern of 'L'/'D' kinds; returns (text, substitutions)."""
    out, cost = [], 0
    for char, kind in zip(chars, kinds):
        if kind == "D":
            if char.isdigit():
                out.append(char)
            elif char in TO_DIGIT:
                out.append(TO_DIGIT[char])
                cost += 1
            else:
                return None
        else:
            if char.isalpha():
                out.append(char)
            elif char in TO_LETTER:
                out.append(TO_LETTER[char])
                cost += 1
            else:
                return None
    return "".join(out), cost


def _fix_state(raw: str) -> tuple[str, int] | None:
    best = None
    options = [[(c, 0)] + [(alt, 1) for alt in STATE_ALTERNATIVES.get(c, "")]
               + ([(TO_LETTER[c], 1)] if c in TO_LETTER else []) for c in raw]
    for (a, ca), (b, cb) in itertools.product(*options):
        if a + b in STATE_CODES and (best is None or ca + cb < best[1]):
            best = (a + b, ca + cb)
    return best


def _parse_exact(s: str) -> tuple[str, int] | None:
    """Best reading of ``s`` as a whole plate, or None."""
    best = None
    for (rto, series), prior in LAYOUTS.items():
        if 2 + rto + series + 4 != len(s):
            continue
        state = _fix_state(s[:2])
        if state is None or (rto == 1 and state[0] != "DL"):
            continue
        rest = _fit(s[2:], "D" * rto + "L" * series + "D" * 4)
        if rest and set(rest[0][rto:rto + series]) & {"I", "O"}:
            continue  # series letters never use I or O (they look like 1 and 0)
        if rest and (best is None or state[1] + rest[1] + prior < best[1]):
            best = (state[0] + rest[0], state[1] + rest[1] + prior)
    if len(s) in (9, 10):  # Bharat series: YY BH NNNN L{1,2}
        fitted = _fit(s, "DDLLDDDD" + "L" * (len(s) - 8))
        if fitted and fitted[0][2:4] == "BH" and (best is None or fitted[1] < best[1]):
            best = fitted
    return best


def parse_plate(text: str, confidence: float) -> Plate | None:
    """Find the most plausible plate inside an OCR string (junk around it is allowed)."""
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    best: Plate | None = None
    best_rank = float("inf")
    for length in range(min(len(s), 11), 6, -1):
        for start in range(len(s) - length + 1):
            parsed = _parse_exact(s[start:start + length])
            if parsed is None:
                continue
            # Dropping surrounding characters (IND badges, frame text) only ranks the
            # candidates of this string; it does not make the plate itself less likely.
            rank = parsed[1] + TRIM_COST * (len(s) - length)
            if rank < best_rank:
                best, best_rank = Plate(parsed[0], parsed[1], confidence), rank
    # More than ~3 corrections means we are inventing a plate out of unrelated text.
    return best if best is not None and best.cost <= 3 else None


def plate_from_lines(lines) -> Plate | None:
    """Best plate in one OCR result; also joins adjacent lines (two-row bike plates)."""
    candidates: list[tuple[str, float]] = [(line.text, line.confidence) for line in lines]
    for upper, lower in zip(lines, lines[1:]):
        candidates.append((upper.text + lower.text, (upper.confidence + lower.confidence) / 2))
    plates = [p for text, conf in candidates if (p := parse_plate(text, conf))]
    return max(plates, key=lambda p: p.score, default=None)


class PlateVoter:
    """Combines one vehicle's plate reads across frames.

    Reads that differ by at most two characters are grouped and the group with the
    highest total confidence wins, so several similar reads (GJ01HP4165, GJ01HP1165)
    outvote one odd read (HP41P4165). Inside the group, reads vote per character, so
    slightly different misreads can still produce the right plate.
    """

    def __init__(self) -> None:
        self.reads: list[Plate] = []

    def add(self, plate: Plate) -> None:
        self.reads.append(plate)

    @property
    def count(self) -> int:
        return len(self.reads)

    def best(self) -> tuple[str, float, int] | None:
        """(plate text, confidence, number of reads that agree), or None."""
        if not self.reads:
            return None
        totals: dict[str, float] = defaultdict(float)
        for read in self.reads:
            totals[read.text] += read.score
        # Each distinct reading collects the votes of every read close to it.
        support = {text: sum(r.score for r in self.reads if _distance(text, r.text) <= 2)
                   for text in totals}
        top = max(totals, key=lambda t: (support[t], totals[t]))
        group = [r for r in self.reads if _distance(top, r.text) <= 2]
        same_length = [r for r in group if len(r.text) == len(top)]
        chars = []
        for i in range(len(top)):
            votes: dict[str, float] = defaultdict(float)
            for read in same_length:
                votes[read.text[i]] += read.score
            chars.append(max(votes, key=votes.get))
        voted = "".join(chars)
        # Only accept the per-character result if it is itself a valid plate.
        text = voted if _parse_exact(voted) == (voted, 0) else top
        agree = sum(1 for r in self.reads if r.text == text)
        confidence = max(r.confidence for r in self.reads if r.text == text) if agree else \
            max(r.confidence for r in same_length)
        return text, confidence, agree


def _distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
