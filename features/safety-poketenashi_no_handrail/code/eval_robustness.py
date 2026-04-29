"""Reproducible event-level robustness eval for safety-poketenashi_no_handrail.

Runs ``NoHandrailPredictor`` on every video listed in ``eval/ground_truth.json``,
classifies emitted ``no_handrail`` alerts against labelled violation windows, and
writes a timestamped report. Auto-updates this feature's CLAUDE.md
"Status & investigation log → A. Current evaluation status" table between the
``<!-- AUTO:section_a -->`` markers.

The ``no_handrail`` rule is **disabled when no handrail polygon is configured**.
This harness therefore reads ``handrail_zones_norm`` from each video's GT entry
(image-normalized [0,1] polygons), denormalizes them to pixel coords using the
video's frame size, and overrides the predictor's ``HandrailDetector._zones``
before processing. Videos with ``handrail_zones_norm: null`` are reported as
``skipped — no polygon configured`` (eval BLOCKED for those).

Usage:
    uv run python features/safety-poketenashi_no_handrail/code/eval_robustness.py
                              # run + write timestamped report + update CLAUDE.md
    uv run python ... --baseline
                              # additionally write eval/robustness_baseline.json
                              # (only if it doesn't already exist)
    uv run python ... --against eval/robustness_baseline.json
                              # diff against the baseline file and print delta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

_FEAT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FEAT / "code"))
sys.path.insert(0, str(_FEAT.parent.parent))  # repo root

from predictor import NoHandrailPredictor  # noqa: E402

_CLAUDE_MD = _FEAT / "CLAUDE.md"
_GT_PATH = _FEAT / "eval" / "ground_truth.json"
_CONFIG = _FEAT / "configs" / "10_inference.yaml"
_SAMPLES = _FEAT / "samples"
_EVAL_DIR = _FEAT / "eval"
_BASELINE = _EVAL_DIR / "robustness_baseline.json"

_SECTION_A_BEGIN = "<!-- AUTO:section_a:begin -->"
_SECTION_A_END = "<!-- AUTO:section_a:end -->"


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------

def _classify_events(events_t: list[float], windows: list[list[float]]) -> dict:
    """TP per matched window; FP per event outside any window; FN per unmatched window."""
    matched: set[int] = set()
    fp = 0
    for t in events_t:
        idx = None
        for i, (s, e) in enumerate(windows):
            if s <= t <= e:
                idx = i
                break
        if idx is None:
            fp += 1
        else:
            matched.add(idx)
    tp = len(matched)
    fn = len(windows) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "matched_windows": sorted(matched)}


def _override_zones(predictor: NoHandrailPredictor, zones_norm: list[list[list[float]]]) -> None:
    """Replace the predictor's handrail-detector zones with the per-video polygon."""
    predictor._rule._zones = zones_norm  # noqa: SLF001 — intentional test-harness reach-in


def _process_video(
    name: str, video_path: Path, windows: list[list[float]], zones_norm: list | None
) -> dict:
    if zones_norm is None:
        return {"skipped": True, "skip_reason": "no polygon configured"}

    predictor = NoHandrailPredictor(_CONFIG)
    _override_zones(predictor, zones_norm)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "could not open video"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 0.0

    events: list[float] = []
    fi = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = predictor.process_frame(frame)
        if "no_handrail" in out.alerts:
            events.append(fi / fps)
        fi += 1
    cap.release()
    elapsed = time.perf_counter() - t0

    cls = _classify_events(events, windows)
    return {
        "duration_s": round(duration, 1),
        "frames": fi,
        "elapsed_s": round(elapsed, 1),
        "events": [round(t, 2) for t in events],
        "first_event_s": round(events[0], 2) if events else None,
        "windows": windows,
        "zones_count": len(zones_norm),
        **cls,
    }


# ---------------------------------------------------------------------------
# Aggregate + report shape
# ---------------------------------------------------------------------------

def _aggregate(per_video: dict[str, dict]) -> dict:
    valid = [v for v in per_video.values() if "error" not in v and not v.get("skipped")]
    tp = sum(v.get("tp", 0) for v in valid)
    fp = sum(v.get("fp", 0) for v in valid)
    fn = sum(v.get("fn", 0) for v in valid)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "evaluable_videos": len(valid),
    }


def _verdict(v: dict) -> str:
    if "error" in v:
        return f"❗ error: {v['error']}"
    if v.get("skipped"):
        return f"⚠️ {v.get('skip_reason', 'skipped')}"
    tp, fp, fn = v["tp"], v["fp"], v["fn"]
    if not v["windows"]:
        return "✅ TN" if fp == 0 else f"❌ FP × {fp}"
    if fp == 0 and fn == 0:
        return f"✅ TP × {tp}"
    if fn > 0 and tp == 0:
        return f"❌ FN × {fn}"
    return f"⚠️ TP {tp} / FP {fp} / FN {fn}"


# ---------------------------------------------------------------------------
# CLAUDE.md auto-update (section A only)
# ---------------------------------------------------------------------------

def _build_section_a_markdown(per_video: dict[str, dict], agg: dict, when: str) -> str:
    lines = [
        f"<!-- last auto-run: {when} -->",
        "",
        f"Evaluable videos: **{agg['evaluable_videos']}**. "
        f"Aggregate: **{agg['tp']} TP, {agg['fp']} FP, {agg['fn']} FN**. "
        f"Precision **{agg['precision']:.3f}**, "
        f"Recall **{agg['recall']:.3f}**, "
        f"F1 **{agg['f1']:.3f}**.",
        "",
        "| Video | Duration | GT windows | Matches (count, first) | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(per_video.keys()):
        v = per_video[name]
        if "error" in v:
            lines.append(f"| `{name}` | — | — | — | ❗ {v['error']} |")
            continue
        if v.get("skipped"):
            lines.append(
                f"| `{name}` | — | (skip) | — | ⚠️ {v.get('skip_reason', 'skipped')} |"
            )
            continue
        windows = (
            ", ".join(f"{s:.0f}–{e:.0f} s" for s, e in v["windows"])
            if v["windows"] else "(none)"
        )
        first = (
            f" (first @ {v['first_event_s']:.1f} s)"
            if v["first_event_s"] is not None else ""
        )
        match_cell = f"{len(v['events'])}{first}"
        lines.append(
            f"| `{name}` | {v['duration_s']:.0f} s | {windows} | {match_cell} | {_verdict(v)} |"
        )
    return "\n".join(lines)


def _update_claude_md(section_a: str) -> bool:
    if not _CLAUDE_MD.exists():
        return False
    text = _CLAUDE_MD.read_text()
    if _SECTION_A_BEGIN in text and _SECTION_A_END in text:
        head, _, rest = text.partition(_SECTION_A_BEGIN)
        _, _, tail = rest.partition(_SECTION_A_END)
        new_text = (
            head
            + _SECTION_A_BEGIN
            + "\n"
            + section_a
            + "\n"
            + _SECTION_A_END
            + tail
        )
    else:
        marker_block = f"\n{_SECTION_A_BEGIN}\n{section_a}\n{_SECTION_A_END}\n"
        new_text = text + "\n" + marker_block
    if new_text != text:
        _CLAUDE_MD.write_text(new_text)
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true",
                        help="Write eval/robustness_baseline.json (only if absent).")
    parser.add_argument("--against", type=Path, default=None,
                        help="Compare current run to a previous report file; print delta.")
    parser.add_argument("--no-update-claude-md", action="store_true",
                        help="Skip the auto-update of CLAUDE.md section A.")
    args = parser.parse_args()

    gt = json.loads(_GT_PATH.read_text())
    videos = gt["videos"]

    per_video: dict[str, dict] = {}
    for name, meta in videos.items():
        path = _SAMPLES / name
        if not path.exists():
            per_video[name] = {"skipped": True, "skip_reason": "video file missing"}
            print(f"  SKIP  {name}  (missing)")
            continue
        zones_norm = meta.get("handrail_zones_norm")
        windows_raw = meta.get("violation_windows")
        if zones_norm is None or windows_raw is None:
            per_video[name] = {
                "skipped": True,
                "skip_reason": "no polygon configured" if zones_norm is None
                else "no violation windows annotated",
            }
            print(f"  SKIP  {name}  ({per_video[name]['skip_reason']})")
            continue
        windows = [list(w) for w in windows_raw]
        print(f"  RUN   {name}  windows={windows}  zones={len(zones_norm)}")
        per_video[name] = _process_video(name, path, windows, zones_norm)
        v = per_video[name]
        if "error" not in v and not v.get("skipped"):
            print(f"        -> {_verdict(v)}  ({v['elapsed_s']}s, {v['frames']} frames)")

    agg = _aggregate(per_video)
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report: dict[str, Any] = {
        "timestamp": when,
        "config_path": str(_CONFIG),
        "ground_truth_path": str(_GT_PATH),
        "aggregate": agg,
        "per_video": per_video,
    }

    out = _EVAL_DIR / f"robustness_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Wrote {out}")
    print(f"  Aggregate: TP={agg['tp']}, FP={agg['fp']}, FN={agg['fn']} "
          f"(over {agg['evaluable_videos']} evaluable videos)")
    print(f"             P={agg['precision']:.3f}  R={agg['recall']:.3f}  F1={agg['f1']:.3f}")

    if args.baseline:
        if _BASELINE.exists():
            print(f"  baseline already exists at {_BASELINE} — refusing to overwrite")
        else:
            _BASELINE.write_text(json.dumps(report, indent=2))
            print(f"  Wrote baseline -> {_BASELINE}")

    if args.against:
        try:
            prev = json.loads(args.against.read_text())
        except Exception as exc:
            print(f"  could not read --against {args.against}: {exc}")
        else:
            prev_agg = prev.get("aggregate", {})
            print(f"\n  delta vs {args.against.name}:")
            for k in ("tp", "fp", "fn", "precision", "recall", "f1"):
                cur = agg.get(k, 0)
                old = prev_agg.get(k, 0)
                d = cur - old if isinstance(cur, (int, float)) else "—"
                sign = "+" if (isinstance(d, (int, float)) and d >= 0) else ""
                print(f"    {k}: {old} -> {cur} ({sign}{d})")

    if not args.no_update_claude_md:
        section_a = _build_section_a_markdown(per_video, agg, when)
        if _update_claude_md(section_a):
            print("  Updated CLAUDE.md section A.")
        else:
            print("  CLAUDE.md unchanged.")


if __name__ == "__main__":
    main()
