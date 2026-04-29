"""Phase 1 — analyze the per-frame debug CSVs and confirm/refine each
hypothesis from the investigation plan.

Reads:
  eval/debug_<video>.csv          (from dump_debug.py)
  eval/robustness_baseline.json   (event timestamps per video)
  eval/ground_truth.json          (gesture windows per video)

Writes:
  eval/failure_mode_analysis.md   (one section per cluster: stats + verdict)

Usage:
    uv run python features/safety-poketenashi_point_and_call/code/analyze_failures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median

_FEAT = Path(__file__).resolve().parent.parent
_EVAL = _FEAT / "eval"


# ---------------------------------------------------------------------------
# CSV loader (small; rows fit in RAM easily for ~10k frames)
# ---------------------------------------------------------------------------

def _load_csv(stem: str) -> list[dict]:
    p = _EVAL / f"debug_{stem}.csv"
    if not p.exists():
        return []
    with p.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("frame",):
            if r.get(k) not in ("", None):
                r[k] = int(r[k])
        for k in ("t", "elbow_angle", "arm_elevation", "azimuth",
                  "wrist_ear_ratio_min", "shoulder_w_px",
                  "score_l_shoulder", "score_l_elbow", "score_l_wrist",
                  "score_r_shoulder", "score_r_elbow", "score_r_wrist",
                  "score_nose", "score_l_ear", "score_r_ear",
                  "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y",
                  "hip_mid_x", "hip_mid_y"):
            if r.get(k) not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None
    return rows


def _baseline_events(name: str) -> list[float]:
    p = _EVAL / "robustness_baseline.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    v = data.get("per_video", {}).get(name, {})
    return v.get("events", [])


def _gt_windows(name: str) -> list[tuple[float, float]]:
    gt = json.loads((_EVAL / "ground_truth.json").read_text())
    return [tuple(w) for w in gt["videos"].get(name, {}).get("gesture_windows", [])]


def _in_any_window(t: float, windows: list[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in windows)


def _classify_event(t: float, windows: list[tuple[float, float]]) -> str:
    return "TP" if _in_any_window(t, windows) else "FP"


# ---------------------------------------------------------------------------
# Discriminator helpers
# ---------------------------------------------------------------------------

def _summary(values: list[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "(none)"
    return (f"n={len(vals)}  min={min(vals):.2f}  "
            f"med={median(vals):.2f}  max={max(vals):.2f}")


def _wrist_speed_window(rows: list[dict], t_center: float, half_window: float = 1.0,
                       fps: float = 30.0) -> dict:
    """Compute wrist (right) instantaneous speed in pixels/sec over rows whose
    `t` is within ±half_window seconds of t_center. Returns simple stats."""
    near = [r for r in rows if abs(r.get("t", -1e9) - t_center) <= half_window]
    speeds = []
    for prev, cur in zip(near, near[1:]):
        if prev.get("wrist_r_x") is None or cur.get("wrist_r_x") is None:
            continue
        dt = max(cur["t"] - prev["t"], 1e-3)
        d = ((cur["wrist_r_x"] - prev["wrist_r_x"]) ** 2
             + (cur["wrist_r_y"] - prev["wrist_r_y"]) ** 2) ** 0.5
        speeds.append(d / dt)
    if not speeds:
        return {"n": 0}
    speeds.sort()
    p25 = speeds[len(speeds) // 4] if len(speeds) >= 4 else speeds[0]
    p75 = speeds[(3 * len(speeds)) // 4] if len(speeds) >= 4 else speeds[-1]
    return {"n": len(speeds), "min": speeds[0], "med": speeds[len(speeds) // 2],
            "max": speeds[-1], "p25": p25, "p75": p75}


# ---------------------------------------------------------------------------
# Cluster analyses
# ---------------------------------------------------------------------------

def analyse_lecture_fps(out: list[str]) -> None:
    """Hypothesis A — lecturer's wrists move continuously; TP gestures show a
    clear "raise → stationary hold → lower" velocity pattern.
    """
    name = "POKETENASHI_anzen_daiichi_lecture.mp4"
    rows = _load_csv("POKETENASHI_anzen_daiichi_lecture")
    tp_rows = _load_csv("05_SHI_point_and_call")
    fp_events = _baseline_events(name)
    tp_events = [t for t in _baseline_events("05_SHI_point_and_call.mp4")
                 if _in_any_window(t, _gt_windows("05_SHI_point_and_call.mp4"))]

    out.append("## Cluster 1 — Lecture FPs (Hypothesis A)\n")
    out.append(f"- Source: `{name}` ({len(fp_events)} events, all FP)")
    out.append(f"- Reference TP: `05_SHI_point_and_call.mp4` ({len(tp_events)} TP events)\n")
    out.append("**Wrist (right) speed over a ±1 s window around each event "
               "(median pixels/sec):**\n")
    out.append("| Cluster | n events | median speed (med ± p25-p75 px/s) |")
    out.append("|---|---|---|")

    fp_meds = []
    for t in fp_events[:10]:  # cap output rows
        s = _wrist_speed_window(rows, t)
        if s.get("n"):
            fp_meds.append(s["med"])
            out.append(f"| FP @ t={t:.1f}s | {s['n']} | {s['med']:.0f} (p25={s.get('p25', 0):.0f} "
                       f"p75={s.get('p75', 0):.0f}) |")
    tp_meds = []
    for t in tp_events:
        s = _wrist_speed_window(tp_rows, t)
        if s.get("n"):
            tp_meds.append(s["med"])
            out.append(f"| TP @ t={t:.1f}s | {s['n']} | {s['med']:.0f} (p25={s.get('p25', 0):.0f} "
                       f"p75={s.get('p75', 0):.0f}) |")
    out.append("")
    if fp_meds and tp_meds:
        out.append(f"**Aggregate**: FP median-of-medians = {median(fp_meds):.0f} px/s; "
                   f"TP median-of-medians = {median(tp_meds):.0f} px/s.\n")
        if median(tp_meds) < median(fp_meds):
            out.append("**Verdict**: ✅ Hypothesis A confirmed — lecturer's wrist "
                       "moves materially faster around event time than a true "
                       "shisa-kanko gesture's stationary hold. Wrist-velocity "
                       "gate (Phase 2 Intervention A) should bisect them. "
                       "Suggested threshold: any frame where ±0.3 s wrist speed "
                       f"exceeds {median(tp_meds) * 1.5:.0f} px/s is NOT in HOLD.")
        else:
            out.append("**Verdict**: ⚠️ Hypothesis A partially supported only — "
                       "speed difference is smaller than expected; combine with "
                       "another signal.")
    else:
        out.append("**Verdict**: not enough data — re-run dump.")
    out.append("")


def analyse_phone_fp(out: list[str]) -> None:
    """Hypothesis B — POKETENASHI's KE phone-on-ear scene fires at t≈82.9s.
    Compare wrist_ear_ratio_min between FP frames (t≈82s) and TP frames in the
    same video's SHI section (t≈190s)."""
    name = "POKETENASHI.mp4"
    rows = _load_csv("POKETENASHI")
    events = _baseline_events(name)
    windows = _gt_windows(name)
    fp_events = [t for t in events if not _in_any_window(t, windows)]
    tp_events = [t for t in events if _in_any_window(t, windows)]

    out.append("## Cluster 2 — POKETENASHI KE phone FP (Hypothesis B)\n")
    out.append(f"- Source: `{name}`")
    out.append(f"- FP events ({len(fp_events)}): " + ", ".join(f"{t:.1f}s" for t in fp_events))
    out.append(f"- TP events ({len(tp_events)}): " + ", ".join(f"{t:.1f}s" for t in tp_events))
    out.append("")
    out.append("**`wrist_ear_ratio_min` at event-frame neighborhoods (median over ±0.3 s):**\n")

    def ratio_at(t: float) -> float | None:
        near = [r["wrist_ear_ratio_min"] for r in rows
                if abs(r.get("t", -1e9) - t) <= 0.3 and r.get("wrist_ear_ratio_min") is not None]
        return median(near) if near else None

    fp_ratios = [ratio_at(t) for t in fp_events]
    tp_ratios = [ratio_at(t) for t in tp_events]

    out.append("| Cluster | t (s) | wrist_ear_ratio_min (median ±0.3 s) |")
    out.append("|---|---|---|")
    for t, r in zip(fp_events, fp_ratios):
        out.append(f"| FP | {t:.1f} | {'—' if r is None else f'{r:.3f}'} |")
    for t, r in zip(tp_events, tp_ratios):
        out.append(f"| TP | {t:.1f} | {'—' if r is None else f'{r:.3f}'} |")
    out.append("")

    fp_clean = [r for r in fp_ratios if r is not None]
    tp_clean = [r for r in tp_ratios if r is not None]
    if fp_clean and tp_clean:
        out.append(f"**Aggregate**: FP median = {median(fp_clean):.3f}, "
                   f"TP median = {median(tp_clean):.3f}.\n")
        if max(fp_clean) < min(tp_clean):
            out.append(f"**Verdict**: ✅ Hypothesis B confirmed — clear gap; "
                       f"raise `min_wrist_ear_distance_ratio` to "
                       f"~{(max(fp_clean) + min(tp_clean)) / 2:.2f} kills the "
                       f"phone-on-ear FP without losing any TP. Could also be "
                       f"keypoint-confidence-gated (Intervention B).")
        else:
            out.append("**Verdict**: ⚠️ overlap exists — single threshold can't "
                       "split cleanly. Two-tier confidence gate (Intervention B) "
                       "is needed.")
    else:
        out.append("**Verdict**: not enough event neighborhoods with valid ratios.")
    out.append("")


def analyse_far_field_fn(out: list[str]) -> None:
    """Hypothesis C — far-field actor produces low keypoint scores."""
    name = "SHI_point_and_call_spkepcmwi.mp4"
    rows = _load_csv("SHI_point_and_call_spkepcmwi")
    windows = _gt_windows(name)

    in_window = [r for r in rows if any(s <= r.get("t", -1) <= e for s, e in windows)]
    if not in_window:
        out.append("## Cluster 3 — Far-field FN (Hypothesis C)\n")
        out.append("**Verdict**: not enough data — re-run dump.\n")
        return

    n_total = len(in_window)
    n_invalid = sum(1 for r in in_window if r.get("label") == "invalid")
    sw_med = median([r["shoulder_w_px"] for r in in_window if r.get("shoulder_w_px")])
    wrist_r_med = median([r["score_r_wrist"] for r in in_window if r.get("score_r_wrist") is not None])
    wrist_l_med = median([r["score_l_wrist"] for r in in_window if r.get("score_l_wrist") is not None])

    out.append("## Cluster 3 — Far-field FN (Hypothesis C)\n")
    out.append(f"- Source: `{name}`, GT windows: {windows}")
    out.append(f"- Frames inside GT window: {n_total}")
    out.append(f"- `label=invalid` rate: {n_invalid}/{n_total} = {100 * n_invalid / n_total:.1f}%")
    out.append(f"- Median shoulder width: {sw_med:.1f} px (frame width 1920)")
    out.append(f"- Median right-wrist score: {wrist_r_med:.3f}")
    out.append(f"- Median left-wrist score: {wrist_l_med:.3f}")
    out.append("")
    if min(wrist_r_med, wrist_l_med) < 0.25:
        out.append("**Verdict**: ✅ Hypothesis C confirmed — wrist scores median "
                   "below the `min_keypoint_score=0.25` threshold inside the "
                   "ground-truth gesture window; rule returns `invalid` on most "
                   f"frames. Pre-pose crop upscaling (Intervention C) targets "
                   f"this directly. Median shoulder width {sw_med:.0f} px on a "
                   f"1920 px frame ≈ {sw_med / 1920 * 100:.1f}% of frame "
                   "width — tiny.")
    else:
        out.append("**Verdict**: ⚠️ wrist scores higher than expected; "
                   "FN must be from a different cause.")
    out.append("")


def analyse_jitter_fps(out: list[str]) -> None:
    """Hypothesis D — pose jitter randomizes labels; investigate within
    lecture FPs (already loaded above)."""
    rows = _load_csv("POKETENASHI_anzen_daiichi_lecture")
    name = "POKETENASHI_anzen_daiichi_lecture.mp4"
    events = _baseline_events(name)
    if not rows or not events:
        return

    out.append("## Cluster 4 — Pose jitter / fast L↔R flips (Hypothesis D)\n")
    out.append("Sample of label transitions in the 30-frame window leading to "
               "each FP event:\n")
    out.append("| FP @ t (s) | last 15 labels |")
    out.append("|---|---|")
    for t in events[:6]:
        seq = []
        for r in rows:
            if 0 <= t - r.get("t", -1e9) <= 0.5:
                lbl = r.get("label", "?")
                short = (lbl.replace("point_", "")
                         .replace("invalid", "x").replace("neutral", "n"))
                seq.append(short)
        out.append(f"| {t:.1f} | {' '.join(seq[-15:])} |")
    out.append("")
    out.append("**Verdict**: read the strings above. If you see L→R→L→R rapid "
               "alternation with no `n` (neutral) between, Hypothesis D is "
               "supported and Intervention D (transition-shape filter) is "
               "warranted. If transitions are smooth (l l l n n r r r), "
               "Intervention D is unnecessary.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out: list[str] = [
        "# Failure-mode analysis (Phase 1)",
        "",
        "Generated by `code/analyze_failures.py` from per-frame debug CSVs "
        "(`code/dump_debug.py`) plus the locked baseline event list "
        "(`eval/robustness_baseline.json`).",
        "",
        "Each section confirms or rejects one hypothesis from the investigation "
        "plan. Confirmed → proceed to Phase 2 intervention. Rejected → revise.",
        "",
    ]
    analyse_lecture_fps(out)
    analyse_phone_fp(out)
    analyse_far_field_fn(out)
    analyse_jitter_fps(out)

    out_path = _EVAL / "failure_mode_analysis.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path.relative_to(_FEAT.parent.parent)}")
    print()
    print("\n".join(out))


if __name__ == "__main__":
    main()
