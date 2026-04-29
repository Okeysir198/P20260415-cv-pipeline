"""Base classes for pose-based behavior rule modules.

Self-contained: do NOT import from another feature's ``code/`` directory.
Mirrors the signatures of ``features/safety-poketenashi/code/_base.py`` so
shared rule patterns work the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RuleResult:
    triggered: bool
    confidence: float  # 0.0-1.0
    behavior: str
    debug_info: dict = field(default_factory=dict)


class PoseRule:
    """Base class for all pose-based behavior rules."""

    behavior: str = ""

    def check(
        self,
        keypoints: np.ndarray,  # (K, 2) pixel coords
        scores: np.ndarray,  # (K,) confidence per keypoint
        frame_buffer: list[np.ndarray] | None = None,
    ) -> RuleResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any temporal state (called between tracks)."""
        pass
