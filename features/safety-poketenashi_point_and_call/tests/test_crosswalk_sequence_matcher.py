"""Tests for the crosswalk sequence matcher."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_FEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FEAT / "code"))

from crosswalk_sequence_matcher import CrosswalkSequenceMatcher  # noqa: E402


def _feed_run(matcher: CrosswalkSequenceMatcher,
              labels_with_counts: list[tuple[str, int]],
              start_t: float = 0.0,
              dt: float = 1.0 / 30) -> dict:
    """Feed runs of repeated labels; returns the last state dict."""
    last = {}
    t = start_t
    for label, n in labels_with_counts:
        for _ in range(n):
            last = matcher.feed(label, t)
            t += dt
    return last


def test_lrf_completes_when_each_run_meets_hold_frames():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=6,
        window_seconds=10.0,
        sequence_modes=["LRF"],
    )
    state = _feed_run(matcher, [
        ("point_left", 6),
        ("point_right", 6),
        ("point_front", 6),
    ])
    assert state["sequence_done"] is True
    assert state["matched_mode"] == "LRF"


def test_short_runs_below_hold_frames_do_not_match():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=6,
        window_seconds=10.0,
        sequence_modes=["LRF"],
    )
    state = _feed_run(matcher, [
        ("point_left", 3),
        ("point_right", 3),
        ("point_front", 3),
    ])
    assert state["sequence_done"] is False


def test_out_of_order_does_not_match_lrf():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=6,
        window_seconds=10.0,
        sequence_modes=["LRF"],
    )
    state = _feed_run(matcher, [
        ("point_front", 6),
        ("point_left", 6),
        ("point_right", 6),
    ])
    assert state["sequence_done"] is False


def test_window_timeout_drops_old_entries():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=4,
        window_seconds=5.0,
        sequence_modes=["LRF"],
    )
    # Long gap between L and R (>5s window).
    state = _feed_run(matcher, [
        ("point_left", 4),
    ], start_t=0.0, dt=0.1)
    assert state["sequence_done"] is False
    # Wait past the window, then start a fresh L+R+F.
    state = _feed_run(matcher, [
        ("point_right", 4),
        ("point_front", 4),
    ], start_t=10.0, dt=0.1)
    # Buffer should have pruned the leading L entirely; only R+F survive
    # which is not the LRF order.
    assert state["sequence_done"] is False


def test_cooldown_blocks_immediate_replay():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=4,
        window_seconds=10.0,
        sequence_modes=["LRF"],
        cooldown_frames=60,
    )
    state = _feed_run(matcher, [
        ("point_left", 4),
        ("point_right", 4),
        ("point_front", 4),
    ])
    assert state["sequence_done"] is True

    # Replay immediately while cooldown is still active: 12 feeds at
    # cooldown_frames=60 means cooldown remains > 0 throughout.
    state = _feed_run(matcher, [
        ("point_left", 4),
        ("point_right", 4),
        ("point_front", 4),
    ], start_t=10.0)
    assert state["sequence_done"] is False
    assert state["cooldown_active"] is True


def test_reset_clears_state():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=4,
        window_seconds=10.0,
        sequence_modes=["LRF"],
    )
    _feed_run(matcher, [("point_left", 4), ("point_right", 4)])
    matcher.reset()
    state = _feed_run(matcher, [("point_left", 4)])
    # After reset only the new L is present; not yet a match.
    assert state["progress"] == ["point_left"]
    assert state["sequence_done"] is False


def test_invalid_mode_letter_raises():
    with pytest.raises(ValueError):
        CrosswalkSequenceMatcher(
            hold_frames=4,
            window_seconds=5.0,
            sequence_modes=["XYZ"],
        )


def test_lr_only_completes_two_step_mode():
    matcher = CrosswalkSequenceMatcher(
        hold_frames=4,
        window_seconds=10.0,
        sequence_modes=["LR"],
    )
    state = _feed_run(matcher, [
        ("point_left", 4),
        ("point_right", 4),
    ])
    assert state["sequence_done"] is True
    assert state["matched_mode"] == "LR"
