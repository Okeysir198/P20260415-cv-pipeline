"""Point-and-call orchestrator (shisa-kanko crosswalk gesture detector).

V1 -- pretrained-only, single-person. The orchestrator dispatches to a
swappable pose backend (DWPose ONNX by default), runs the per-frame
``PointingDirectionDetector``, and feeds frame labels into a
``CrosswalkSequenceMatcher`` per track. Smoke-test CLI exercises the
full path on sample images and writes ``eval/orchestrator_smoke_test.json``.

Run smoke test:
    uv run features/safety-poketenashi_point_and_call/code/orchestrator.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

REPO = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).parent

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_CODE_DIR))

from utils.config import load_config  # noqa: E402
from utils.viz import VizStyle, annotate_keypoints, classification_banner  # noqa: E402

from _base import RuleResult  # noqa: E402
from _zone import ZoneFSM, make_gate  # noqa: E402
from crosswalk_sequence_matcher import CrosswalkSequenceMatcher  # noqa: E402
from pointing_direction_detector import PointingDirectionDetector  # noqa: E402
from pose_backend import build_pose_backend  # noqa: E402

# COCO-17 skeleton edges for visualisation.
_SKELETON_17 = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

_DIRECTION_RULE_KEYS = {
    "elbow_angle_min_deg",
    "arm_elevation_max_deg",
    "front_half_angle_deg",
    "side_half_angle_deg",
    "min_keypoint_score",
    "min_wrist_ear_distance_ratio",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PersonBehavior:
    track_id: int
    keypoints: np.ndarray  # (K, 2)
    kp_scores: np.ndarray  # (K,)
    direction_label: str
    direction_result: RuleResult
    sequence_state: dict


@dataclass
class OrchestratorResult:
    alerts: list[str]
    persons: list[PersonBehavior]
    latency_ms: float
    current_label: str | None
    sequence_progress: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PointAndCallOrchestrator:
    """Single-person rule-based shisa-kanko detector (v1)."""

    def __init__(
        self,
        config_path: str | Path,
        pose_backend_override: str | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._cfg = load_config(self._config_path)

        pose_cfg = dict(self._cfg.get("pose", {}))
        if pose_backend_override:
            pose_cfg["backend"] = pose_backend_override
        pose_cfg.setdefault("_config_dir", str(self._config_path.parent))
        self._pose = build_pose_backend(pose_cfg)
        self._pose_backend_name = str(pose_cfg.get("backend"))

        rules_cfg = self._cfg.get("pose_rules", {}) or {}
        pac_cfg = rules_cfg.get("point_and_call", {}) or {}
        rule_kwargs = {k: v for k, v in pac_cfg.items() if k in _DIRECTION_RULE_KEYS}
        self._direction_rule = PointingDirectionDetector(**rule_kwargs)

        # Stationary-actor gate: a true shisa-kanko gesture is performed at the
        # curb (worker stops, points, then crosses). Walking actors with briefly
        # extended arms (the SHI/NA "bad example" scenes) look identical
        # frame-to-frame. Forcing label -> neutral when hips translate faster
        # than this threshold removes that false-positive class.
        self._max_body_speed_px_per_sec = float(pac_cfg.get("max_body_speed_px_per_sec", 0.0))
        # Per-track sliding window of (timestamp, hip_xy) for smoothed speed.
        from collections import deque
        self._hip_history: dict[int, deque] = {}
        self._hip_window_seconds = 1.5

        # Sequence params live under pose_rules.point_and_call: in 10_inference.yaml;
        # cooldown_frames lives under alerts:.
        alerts_cfg = self._cfg.get("alerts", {}) or {}
        self._seq_kwargs = {
            "hold_frames": int(pac_cfg.get("hold_frames", 5)),
            "window_seconds": float(pac_cfg.get("window_seconds", 8.0)),
            "sequence_modes": list(
                pac_cfg.get("sequence_modes", ["LRF", "LFR", "RLF", "RFL", "FLR", "FRL"])
            ),
            "cooldown_frames": int(alerts_cfg.get("cooldown_frames", 90)),
            "min_distinct_directions": int(pac_cfg.get("min_distinct_directions", 0)),
            "require_rest_between_directions": bool(
                pac_cfg.get("require_rest_between_directions", False)
            ),
            "min_rest_frames": int(pac_cfg.get("min_rest_frames", 3)),
            "max_hold_frames": int(pac_cfg.get("max_hold_frames", 0)),
        }
        # Per-track matcher. v1 uses a single track id 0.
        self._matchers: dict[int, CrosswalkSequenceMatcher] = {}
        self._frame_index = 0
        self._t0_wall: float | None = None

        # Zone-FSM: gates the rule on a per-track APPROACHING-state. Default
        # config carries empty polygons (`approach_zone: []`, `cross_zone: []`)
        # → FSM stays APPROACHING forever → rule armed always (current behavior).
        # When polygons are configured, the rule fires only while the tracked
        # actor's foot point is in the approach polygon.
        self._approach_zone_norm: list[list[float]] = list(
            self._cfg.get("approach_zone", []) or []
        )
        self._cross_zone_norm: list[list[float]] = list(
            self._cfg.get("cross_zone", []) or []
        )
        self._zone_fsms: dict[int, ZoneFSM] = {}

    # ------------------------------------------------------------------
    # Per-track matcher (lazy)
    # ------------------------------------------------------------------

    def _get_matcher(self, track_id: int) -> CrosswalkSequenceMatcher:
        m = self._matchers.get(track_id)
        if m is None:
            m = CrosswalkSequenceMatcher(**self._seq_kwargs)
            self._matchers[track_id] = m
        return m

    def _get_zone_fsm(self, track_id: int) -> ZoneFSM:
        f = self._zone_fsms.get(track_id)
        if f is None:
            f = ZoneFSM(make_gate(self._approach_zone_norm, self._cross_zone_norm))
            self._zone_fsms[track_id] = f
        return f

    def set_zones(self, approach_norm: list[list[float]] | None,
                  cross_norm: list[list[float]] | None) -> None:
        """Override the YAML-loaded polygons at runtime.

        Used by the eval harness to apply per-video annotations from
        `eval/ground_truth.json`. Resets per-track FSMs so a new gate takes
        effect on the next frame.
        """
        self._approach_zone_norm = list(approach_norm or [])
        self._cross_zone_norm = list(cross_norm or [])
        self._zone_fsms.clear()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image_bgr: np.ndarray,
        timestamp: float | None = None,
    ) -> OrchestratorResult:
        t0 = time.perf_counter()
        if self._t0_wall is None:
            self._t0_wall = t0
        if timestamp is None:
            timestamp = t0 - self._t0_wall

        pose_samples = self._pose(image_bgr)
        # v1: single-person matcher. When multiple people are visible (forklift
        # driver + worker, etc.) feeding all their labels into track_id=0
        # produces label-flicker that fakes a "multi-direction sequence". Keep
        # only the largest-area person box so the matcher sees one consistent
        # actor.
        if len(pose_samples) > 1:
            def _box_area(s):
                b = s[2]
                return float((b[2] - b[0]) * (b[3] - b[1]))
            pose_samples = [max(pose_samples, key=_box_area)]

        persons: list[PersonBehavior] = []
        alerts: list[str] = []
        current_label: str | None = None
        sequence_progress: list[str] = []

        h, w = image_bgr.shape[:2]
        for kpts, scores, box in pose_samples:
            res = self._direction_rule.check(kpts, scores)
            label = str(res.debug_info.get("label", "invalid"))

            track_id = 0  # v1: single-person.

            # Zone-FSM update: foot point at bottom-center of the person box.
            # Stays in APPROACHING (rule armed) when no approach polygon is
            # configured — backward-compat for sample videos without polygons.
            foot_xy = (float((box[0] + box[2]) * 0.5), float(box[3]))
            fsm = self._get_zone_fsm(track_id)
            zone_state = fsm.update(foot_xy, w, h)
            res.debug_info["zone_state"] = zone_state
            if not fsm.armed:
                # Rule disarmed by the zone gate (e.g. lecturer is not in any
                # configured approach polygon). Skip the matcher.feed() so no
                # accumulation toward a match.
                persons.append(
                    PersonBehavior(
                        track_id=track_id,
                        keypoints=np.asarray(kpts, dtype=np.float32),
                        kp_scores=np.asarray(scores, dtype=np.float32),
                        direction_label="zone_disarmed",
                        direction_result=res,
                        sequence_state={"zone_state": zone_state, "armed": False,
                                          "sequence_done": False, "progress": []},
                    )
                )
                if current_label is None:
                    current_label = "zone_disarmed"
                continue

            # Stationary-actor gate: suppress pointing labels while the actor's
            # hips are translating fast (i.e. they're walking, not standing at
            # the curb performing the gesture). Speed is smoothed over a
            # 0.5 s sliding window so single-frame pose jitter doesn't fake
            # a "stopped" instant inside a walking sequence.
            from collections import deque
            hip_mid = 0.5 * (np.asarray(kpts[11]) + np.asarray(kpts[12]))
            hist = self._hip_history.setdefault(track_id, deque())
            hist.append((float(timestamp), hip_mid))
            cutoff = float(timestamp) - self._hip_window_seconds
            while hist and hist[0][0] < cutoff:
                hist.popleft()
            if self._max_body_speed_px_per_sec > 0.0 and label.startswith("point_") and len(hist) >= 2:
                t0, xy0 = hist[0]
                t1, xy1 = hist[-1]
                dt = max(t1 - t0, 1e-3)
                avg_speed = float(np.linalg.norm(xy1 - xy0)) / dt
                if avg_speed > self._max_body_speed_px_per_sec:
                    label = "neutral"
                    res.debug_info["label"] = "neutral"
                    res.debug_info["suppressed_by"] = f"walking ({avg_speed:.0f}px/s)"
            matcher = self._get_matcher(track_id)
            seq_state = matcher.feed(label, timestamp)

            if seq_state.get("sequence_done"):
                if "point_and_call_done" not in alerts:
                    alerts.append("point_and_call_done")

            if current_label is None:
                current_label = label
                sequence_progress = list(seq_state.get("progress", []))

            persons.append(
                PersonBehavior(
                    track_id=track_id,
                    keypoints=np.asarray(kpts, dtype=np.float32),
                    kp_scores=np.asarray(scores, dtype=np.float32),
                    direction_label=label,
                    direction_result=res,
                    sequence_state=seq_state,
                )
            )

        self._frame_index += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OrchestratorResult(
            alerts=alerts,
            persons=persons,
            latency_ms=latency_ms,
            current_label=current_label,
            sequence_progress=sequence_progress,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(self, image_bgr: np.ndarray, result: OrchestratorResult) -> np.ndarray:
        conf_th = 0.3
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        body_style = VizStyle(
            kpt_visibility_threshold=conf_th,
            skeleton_color_rgb=(255, 200, 0),
        )

        for person in result.persons:
            rgb = annotate_keypoints(
                rgb,
                person.keypoints,
                skeleton_edges=_SKELETON_17,
                confidence=person.kp_scores,
                style=body_style,
                color=sv.Color(r=0, g=0, b=255),
            )

        # Top banner: current label + sequence progress.
        progress_str = (
            " -> ".join(p.replace("point_", "") for p in result.sequence_progress)
            if result.sequence_progress
            else "-"
        )
        banner_text = (
            f"label={result.current_label or 'no_person'} | "
            f"progress={progress_str}"
        )
        if result.alerts:
            banner_text += "  | ALERT: " + ",".join(result.alerts)
        rgb = classification_banner(
            rgb,
            banner_text,
            style=VizStyle(banner_height=24, banner_text_scale=0.5),
            position="overlay_top",
            bg_color_rgb=(30, 30, 30),
            text_color_rgb=(255, 255, 255) if not result.alerts else (255, 80, 80),
        )

        # Bottom: latency.
        rgb = classification_banner(
            rgb,
            f"{result.latency_ms:.1f}ms ({self._pose_backend_name})",
            style=VizStyle(banner_height=18, banner_text_scale=0.4),
            position="bottom",
            bg_color_rgb=(20, 20, 20),
            text_color_rgb=(160, 160, 160),
        )

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Smoke test CLI
# ---------------------------------------------------------------------------

def _resolve_samples_dir(feat_dir: Path) -> Path:
    own = feat_dir / "samples"
    if own.exists() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                            for p in own.iterdir()):
        return own
    fallback = REPO / "features" / "safety-poketenashi" / "samples"
    return fallback


def _smoke_test(pose_backend_override: str | None) -> None:
    feat = REPO / "features" / "safety-poketenashi_point_and_call"
    config_path = feat / "configs" / "10_inference.yaml"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = _resolve_samples_dir(feat)
    image_paths = sorted(
        p for p in samples_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        print(f"No sample images found in {samples_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Samples dir : {samples_dir} ({len(image_paths)} images)")
    print(f"Config      : {config_path}")
    if pose_backend_override:
        print(f"Pose backend override: {pose_backend_override}")

    orchestrator = PointAndCallOrchestrator(
        config_path, pose_backend_override=pose_backend_override
    )

    print(f"\n{'Image':<32} {'Label':<14} {'Persons':>7} {'ms':>7}")
    print("-" * 70)

    results: list[dict] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip {img_path.name}: unreadable")
            continue

        out = orchestrator.process_frame(img)

        persons_serialised = [
            {
                "track_id": p.track_id,
                "label": p.direction_label,
                "confidence": float(p.direction_result.confidence),
                "debug": {
                    k: v for k, v in p.direction_result.debug_info.items()
                },
                "sequence_state": p.sequence_state,
            }
            for p in out.persons
        ]
        results.append(
            {
                "image": img_path.name,
                "current_label": out.current_label,
                "alerts": out.alerts,
                "sequence_progress": out.sequence_progress,
                "persons": persons_serialised,
                "latency_ms": round(out.latency_ms, 2),
            }
        )

        print(
            f"  {img_path.name:<30} {str(out.current_label or '-'):<14} "
            f"{len(out.persons):>7} {out.latency_ms:>7.1f}"
        )

        annotated = orchestrator.draw(img, out)
        cv2.imwrite(str(eval_dir / f"smoke_{img_path.name}"), annotated)

    report_path = eval_dir / "orchestrator_smoke_test.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results -> {report_path}")
    print(f"Annotated images       -> {eval_dir}/smoke_*.jpg")


def _video_test(video_path: Path, pose_backend_override: str | None) -> None:
    feat = REPO / "features" / "safety-poketenashi_point_and_call"
    config_path = feat / "configs" / "10_inference.yaml"
    eval_dir = feat / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video       : {video_path}  ({width}x{height} @ {fps:.2f} fps, {n_frames} frames)")
    print(f"Config      : {config_path}")
    if pose_backend_override:
        print(f"Pose backend override: {pose_backend_override}")

    orchestrator = PointAndCallOrchestrator(
        config_path, pose_backend_override=pose_backend_override
    )

    out_mp4 = eval_dir / f"smoke_{video_path.stem}.mp4"
    out_json = eval_dir / f"smoke_{video_path.stem}.json"

    # OpenCV's libx264 is unavailable on this host; pipe BGR frames straight to
    # the ffmpeg binary instead.
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )

    timeline: list[dict] = []
    label_hist: dict[str, int] = {}
    first_done_frame: int | None = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        out = orchestrator.process_frame(frame, timestamp=timestamp)

        label = out.current_label or "no_person"
        label_hist[label] = label_hist.get(label, 0) + 1
        if first_done_frame is None and "point_and_call_done" in out.alerts:
            first_done_frame = frame_idx

        timeline.append(
            {
                "frame": frame_idx,
                "t": round(timestamp, 3),
                "label": label,
                "progress": out.sequence_progress,
                "alerts": out.alerts,
                "latency_ms": round(out.latency_ms, 2),
            }
        )

        ff.stdin.write(orchestrator.draw(frame, out).tobytes())

        if frame_idx % 60 == 0:
            print(
                f"  f={frame_idx:4d}  t={timestamp:5.2f}s  label={label:<13}  "
                f"progress={[p.replace('point_', '') for p in out.sequence_progress]}"
                + (f"  ALERT={out.alerts}" if out.alerts else "")
            )
        frame_idx += 1

    cap.release()
    ff.stdin.close()
    ff.wait()
    out_json.write_text(json.dumps(timeline, indent=2))

    print(f"\nFrames processed         : {frame_idx}")
    print(f"Label histogram          : {label_hist}")
    print(
        "First point_and_call_done: "
        + (f"frame {first_done_frame} (t={first_done_frame / fps:.2f}s)"
           if first_done_frame is not None else "no match")
    )
    print(f"Annotated video          : {out_mp4}")
    print(f"Per-frame timeline       : {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-and-call orchestrator (shisa-kanko)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test on sample images in samples/")
    parser.add_argument("--video", type=Path, default=None,
                        help="Run on a video file; writes annotated mp4 + per-frame JSON to eval/")
    parser.add_argument(
        "--pose-backend",
        choices=["dwpose_onnx", "rtmpose", "mediapipe", "hf_keypoint"],
        default=None,
        help="Override the pose backend chosen by 10_inference.yaml",
    )
    args = parser.parse_args()
    if args.video is not None:
        _video_test(args.video, pose_backend_override=args.pose_backend)
    elif args.smoke_test:
        _smoke_test(pose_backend_override=args.pose_backend)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
