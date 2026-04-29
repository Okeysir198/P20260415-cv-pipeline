"""Temporal matcher for the shisa-kanko crosswalk sequence.

Receives per-frame direction labels (``point_left`` / ``point_right`` /
``point_front`` / ``neutral`` / ``invalid``) and decides when an ordered
sequence has been completed within a sliding time window. After a
successful match the matcher enters a cooldown so the same gesture
isn't re-emitted on every subsequent frame.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

_VALID_DIRECTIONS = {"point_left", "point_right", "point_front"}

# Sequence-mode strings (e.g. "LR", "RLF") -> ordered list of direction labels.
_LETTER_TO_LABEL: dict[str, str] = {
    "L": "point_left",
    "R": "point_right",
    "F": "point_front",
}


def _expand_mode(mode: str) -> list[str]:
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"sequence mode must be a non-empty string, got {mode!r}")
    out: list[str] = []
    for ch in mode.upper():
        if ch not in _LETTER_TO_LABEL:
            raise ValueError(
                f"sequence mode {mode!r} has unknown direction letter {ch!r}; "
                f"expected one of L/R/F"
            )
        out.append(_LETTER_TO_LABEL[ch])
    return out


class CrosswalkSequenceMatcher:
    """Sliding-window run-length matcher for direction sequences."""

    def __init__(
        self,
        hold_frames: int,
        window_seconds: float,
        sequence_modes: Iterable[str],
        cooldown_frames: int = 90,
        min_distinct_directions: int = 0,
        require_rest_between_directions: bool = False,
        min_rest_frames: int = 3,
    ) -> None:
        if hold_frames < 1:
            raise ValueError(f"hold_frames must be >= 1, got {hold_frames}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if not 0 <= min_distinct_directions <= 3:
            raise ValueError(
                f"min_distinct_directions must be 0..3, got {min_distinct_directions}"
            )
        modes = list(sequence_modes)
        if not modes and min_distinct_directions == 0:
            raise ValueError(
                "either sequence_modes must be non-empty or min_distinct_directions must be > 0"
            )

        self._hold_frames = int(hold_frames)
        self._window_seconds = float(window_seconds)
        self._cooldown_frames = int(cooldown_frames)
        self._min_distinct_directions = int(min_distinct_directions)
        self._require_rest = bool(require_rest_between_directions)
        self._min_rest_frames = int(min_rest_frames)
        self._modes_expanded: list[tuple[str, list[str]]] = [
            (m, _expand_mode(m)) for m in modes
        ]

        # (timestamp, label) entries.
        self._buf: deque[tuple[float, str]] = deque()
        self._cooldown_left = 0
        self._last_match_mode: str | None = None

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._buf.clear()
        self._cooldown_left = 0
        self._last_match_mode = None

    def feed(self, label: str, timestamp: float) -> dict:
        """Record one frame's label and return the current match state."""
        self._buf.append((float(timestamp), str(label)))
        self._prune(float(timestamp))

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        progress = self._dedup_valid_runs()

        sequence_done = False
        matched_mode: str | None = None
        if self._cooldown_left == 0:
            matched_mode = self._match_against_modes(progress)
            if matched_mode is not None:
                sequence_done = True
                self._cooldown_left = self._cooldown_frames
                self._last_match_mode = matched_mode
                # Clear buffer so a subsequent run starts fresh.
                self._buf.clear()

        return {
            "sequence_done": sequence_done,
            "missing_directions": False,  # set externally by orchestrator
            "matched_mode": matched_mode,
            "progress": progress,
            "cooldown_active": self._cooldown_left > 0,
        }

    # ------------------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def _dedup_valid_runs(self) -> list[str]:
        """Collapse consecutive same-label frames into runs; keep ones with
        run length >= hold_frames AND label in the valid direction set;
        deduplicate by first occurrence per direction."""
        runs: list[tuple[str, int]] = []
        cur_label: str | None = None
        cur_count = 0
        for _ts, label in self._buf:
            if label == cur_label:
                cur_count += 1
            else:
                if cur_label is not None:
                    runs.append((cur_label, cur_count))
                cur_label = label
                cur_count = 1
        if cur_label is not None:
            runs.append((cur_label, cur_count))

        seen: set[str] = set()
        progress: list[str] = []
        # Initially treat the buffer as if rest preceded the first direction.
        # After each accepted direction, rest_ok flips False until a sufficient
        # `neutral` run is seen (Intervention A' — Phase 2). Without this,
        # a presenter holding pose-1 sustained then pivoting to pose-2 would
        # accumulate two distinct directions without ever lowering the arm.
        rest_ok = True
        for label, count in runs:
            if self._require_rest and label == "neutral":
                if count >= self._min_rest_frames:
                    rest_ok = True
                continue
            if count < self._hold_frames:
                continue
            if label not in _VALID_DIRECTIONS:
                continue
            if label in seen:
                continue
            if self._require_rest and progress and not rest_ok:
                # New direction without a "rest" event since the last one.
                # Skip — likely a presenter sliding pose-1 → pose-2 without
                # returning to arms-at-sides.
                continue
            seen.add(label)
            progress.append(label)
            rest_ok = False
        return progress

    def _match_against_modes(self, progress: list[str]) -> str | None:
        # Strict ordered-permutation match (existing logic).
        for mode_str, expected in self._modes_expanded:
            if progress[: len(expected)] == expected:
                return mode_str
        # Robust fallback: ≥ N distinct directions of any kind in the window.
        if self._min_distinct_directions > 0:
            distinct = sum(1 for d in _VALID_DIRECTIONS if d in progress)
            if distinct >= self._min_distinct_directions:
                return f"ANY{distinct}"
        return None
