"""Per-track zone-FSM for point-and-call deployment gating.

A real shisa-kanko gesture happens at the curb of a crosswalk:
worker enters the **approach** zone, performs the gesture, then enters the
**cross** zone (the road itself), then exits. The pose-only rule fires on any
arm-extended pose by anyone in frame — including a lecturer at a podium —
because it has no notion of *where* the actor is. This module adds that
notion.

Each tracked actor (track_id) gets a small state machine driven by the
foot-point of their detected box against image-normalized polygons:

    IDLE        →  (foot in approach polygon)             →  APPROACHING
    APPROACHING →  (foot leaves approach without crossing) →  IDLE
    APPROACHING →  (foot in cross polygon)                 →  CROSSING
    CROSSING    →  (foot leaves cross polygon)             →  DONE
    DONE        →  (foot back in approach polygon)         →  APPROACHING

Only `APPROACHING` arms the per-frame rule (matcher.feed). The other states
discard labels — so a lecturer (never in any approach polygon) never feeds
the matcher and never produces a `point_and_call_done` event.

Backward compatibility: if `approach_norm` is empty/None, `in_approach`
returns True for every point — the FSM stays APPROACHING forever, which
matches the pre-FSM "always armed" behaviour. Existing TP videos without
configured polygons still work.

Polygon eval uses `matplotlib.path.Path.contains_point` to match the pattern
already used by `features/access-zone_intrusion/code/zone_intrusion.py`. We
duplicate the helper here rather than import (project rule: a feature's
`code/` may not import another feature's `code/`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as _MplPath


_STATES = ("IDLE", "APPROACHING", "CROSSING", "DONE")


def _denorm(poly_norm: list[list[float]], w: int, h: int) -> np.ndarray:
    return np.array([[p[0] * w, p[1] * h] for p in poly_norm], dtype=np.float32)


@dataclass
class _Gate:
    approach_norm: list[list[float]]
    cross_norm: list[list[float]]

    def has_approach(self) -> bool:
        return bool(self.approach_norm)

    def has_cross(self) -> bool:
        return bool(self.cross_norm)

    def in_approach(self, foot_xy: tuple[float, float], w: int, h: int) -> bool:
        if not self.approach_norm:
            return True   # no polygon configured ⇒ treat every point as "in"
        path = _MplPath(_denorm(self.approach_norm, w, h))
        return bool(path.contains_point(foot_xy))

    def in_cross(self, foot_xy: tuple[float, float], w: int, h: int) -> bool:
        if not self.cross_norm:
            return False  # no polygon configured ⇒ CROSSING state unreachable
        path = _MplPath(_denorm(self.cross_norm, w, h))
        return bool(path.contains_point(foot_xy))


class ZoneFSM:
    """Per-track state machine. Instantiate one per track_id."""

    def __init__(self, gate: _Gate) -> None:
        self._gate = gate
        self.state: str = "IDLE"

    def update(self, foot_xy: tuple[float, float], w: int, h: int) -> str:
        in_approach = self._gate.in_approach(foot_xy, w, h)
        in_cross = self._gate.in_cross(foot_xy, w, h)

        if self.state == "IDLE":
            if in_approach:
                self.state = "APPROACHING"
        elif self.state == "APPROACHING":
            if in_cross:
                self.state = "CROSSING"
            elif not in_approach:
                self.state = "IDLE"
        elif self.state == "CROSSING":
            if not in_cross:
                self.state = "DONE"
        elif self.state == "DONE":
            if in_approach and not in_cross:
                self.state = "APPROACHING"
        return self.state

    @property
    def armed(self) -> bool:
        """Rule should fire only while in APPROACHING."""
        return self.state == "APPROACHING"


def make_gate(approach_norm: list[list[float]] | None,
              cross_norm: list[list[float]] | None) -> _Gate:
    return _Gate(approach_norm=approach_norm or [], cross_norm=cross_norm or [])
