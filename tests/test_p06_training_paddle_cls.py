"""Test: Paddle backend — Classification (PP-LCNet).

Status: skipped in v1. The paddle backend currently supports detection only
(PicoDet / PP-YOLOE via ``core/p06_paddle/train.py``). Adding PaddleClas requires
a thin driver that mirrors ``_train_detection`` against PaddleClas's Engine —
roughly half a day of integration work.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all  # noqa: E402


def test_paddle_cls_full_chain():
    print("SKIP: paddle backend v1 supports detection only — "
          "PaddleClas driver pending (track in core/p06_paddle/train.py::_TASK_DISPATCH)")
    return


if __name__ == "__main__":
    run_all(
        [("paddle_cls_full_chain", test_paddle_cls_full_chain)],
        title="Paddle Backend — Classification (skipped, v1=detection only)",
    )
