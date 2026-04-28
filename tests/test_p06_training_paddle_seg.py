"""Test: Paddle backend — Segmentation (PP-LiteSeg). Skipped in v1.

The paddle backend currently supports detection only. PaddleSeg driver pending.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402


def test_paddle_seg_full_chain():
    print("SKIP: paddle backend v1 supports detection only — "
          "PaddleSeg driver pending (track in core/p06_paddle/train.py::_TASK_DISPATCH)")
    return


if __name__ == "__main__":
    run_all(
        [("paddle_seg_full_chain", test_paddle_seg_full_chain)],
        title="Paddle Backend — Segmentation (skipped, v1=detection only)",
    )
