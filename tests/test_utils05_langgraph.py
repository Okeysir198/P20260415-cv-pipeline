"""Test 05: Utils — langgraph_common reducers and state helpers."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.langgraph_common import (
    get_batch_paths,
    get_batch_range,
    list_append_reducer,
    make_serialisable,
    replace_reducer,
    should_continue,
)


def test_replace_reducer():
    assert replace_reducer("old", "new") == "new"
    assert replace_reducer(None, 42) == 42
    assert replace_reducer([1, 2], [3]) == [3]


def test_list_append_reducer_replaces():
    # Contract: replaces (nodes manage append themselves)
    assert list_append_reducer([1, 2], [3, 4]) == [3, 4]
    assert list_append_reducer([], [1]) == [1]


def _state(total_images: int, batch_idx: int, batch_size: int, use_sampled: bool = False) -> dict:
    train = [Path(f"a/train/{i}.jpg") for i in range(total_images // 2)]
    val = [Path(f"a/val/{i}.jpg") for i in range(total_images - len(train))]
    key = "sampled_paths" if use_sampled else "image_paths"
    total_batches = (total_images + batch_size - 1) // batch_size
    return {
        key: {"train": train, "val": val},
        "current_batch_idx": batch_idx,
        "batch_size": batch_size,
        "total_batches": total_batches,
    }


def test_get_batch_range_first_batch():
    state = _state(10, batch_idx=0, batch_size=4)
    assert get_batch_range(state) == (0, 4)


def test_get_batch_range_last_partial_batch():
    state = _state(10, batch_idx=2, batch_size=4)
    # start=8, end=min(12,10)=10
    assert get_batch_range(state) == (8, 10)


def test_get_batch_range_prefers_sampled_paths():
    state = _state(10, batch_idx=0, batch_size=4, use_sampled=True)
    assert "sampled_paths" in state
    assert get_batch_range(state) == (0, 4)


def test_get_batch_paths_returns_split_path_tuples():
    state = _state(10, batch_idx=1, batch_size=4)
    batch = get_batch_paths(state)
    assert len(batch) == 4
    for split, p in batch:
        assert split in {"train", "val"}
        assert isinstance(p, Path)


def test_should_continue():
    state = _state(10, batch_idx=1, batch_size=4)  # total_batches=3
    assert should_continue(state) == "continue"
    state["current_batch_idx"] = 3
    assert should_continue(state) == "aggregate"


def test_make_serialisable_numpy():
    payload = {
        "i": np.int64(7),
        "f": np.float32(1.5),
        "arr": np.array([1, 2, 3]),
        "nested": {"x": [np.int32(1), np.float64(2.5)]},
        "tuple": (np.int16(0), "s"),
        "plain": 42,
    }
    result = make_serialisable(payload)

    import json
    # Must round-trip through JSON (proves pure-python types)
    dumped = json.dumps(result)
    loaded = json.loads(dumped)

    assert loaded["i"] == 7
    assert loaded["f"] == 1.5
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["nested"]["x"] == [1, 2.5]
    assert loaded["tuple"] == [0, "s"]
    assert loaded["plain"] == 42


if __name__ == "__main__":
    run_all([
        ("replace_reducer", test_replace_reducer),
        ("list_append_reducer", test_list_append_reducer_replaces),
        ("get_batch_range_first", test_get_batch_range_first_batch),
        ("get_batch_range_partial", test_get_batch_range_last_partial_batch),
        ("get_batch_range_sampled", test_get_batch_range_prefers_sampled_paths),
        ("get_batch_paths", test_get_batch_paths_returns_split_path_tuples),
        ("should_continue", test_should_continue),
        ("make_serialisable_numpy", test_make_serialisable_numpy),
    ], title="Test utils05: langgraph_common")
