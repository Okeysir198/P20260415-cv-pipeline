"""Video inference pipeline with frame-level alert logic.

Processes video files frame-by-frame using :class:`DetectionPredictor`,
accumulates per-class violation counters, and triggers alerts based
on configurable thresholds and confirmation windows.
"""

import contextlib
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p10_inference.predictor import DetectionPredictor

import supervision as sv

import core.p10_inference.supervision_bridge as _sv_bridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _H264Writer:
    """Write BGR frames directly to an H.264 MP4 using PyAV (web-ready, single pass).

    Uses ``movflags=+faststart`` so the ``moov`` atom appears before ``mdat``,
    which is required for browser streaming. Falls back from ``libx264`` to the
    generic ``h264`` encoder alias if the former is unavailable.
    """

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self._container = av.open(
            str(path), mode="w", options={"movflags": "+faststart"}
        )
        codec = "libx264"
        try:
            av.codec.Codec(codec, "w")
        except Exception:
            codec = "h264"
        self._stream = self._container.add_stream(codec, rate=round(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": "23", "preset": "fast"}
        self._frame_idx = 0

    def write_frame(self, bgr: np.ndarray) -> None:
        rgb = bgr[:, :, ::-1]
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        av_frame.pts = self._frame_idx
        self._frame_idx += 1
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    def close(self) -> None:
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        self._container.close()


# ---------------------------------------------------------------------------
# Neutral alert defaults — per-feature overrides live in
# features/<name>/configs/10_inference.yaml (load via `load_alert_config`).
# Empty per-class maps mean: use the predictor's global conf and fire
# immediately (no multi-frame confirmation).
# ---------------------------------------------------------------------------

_NEUTRAL_ALERT_DEFAULTS: dict[str, Any] = {
    "confidence_thresholds": {},   # {class_name: float} — empty ⇒ use global conf
    "frame_windows": {},           # {class_name: int}   — empty ⇒ fire immediately
    "window_ratio": 0.8,
    "cooldown_frames": 90,
}


def load_alert_config(
    path: str | Path | None,
) -> dict[str, Any]:
    """Load the ``alerts:`` block from a feature's ``10_inference.yaml``.

    Merges file contents over :data:`_NEUTRAL_ALERT_DEFAULTS`. Returns the
    neutral defaults when *path* is ``None`` or the file is missing the
    ``alerts`` key.

    Args:
        path: Path to a YAML file with an ``alerts:`` top-level key.

    Returns:
        Alert-config dict suitable for passing to :class:`VideoProcessor`.
    """
    merged: dict[str, Any] = {
        "confidence_thresholds": dict(_NEUTRAL_ALERT_DEFAULTS["confidence_thresholds"]),
        "frame_windows": dict(_NEUTRAL_ALERT_DEFAULTS["frame_windows"]),
        "window_ratio": _NEUTRAL_ALERT_DEFAULTS["window_ratio"],
        "cooldown_frames": _NEUTRAL_ALERT_DEFAULTS["cooldown_frames"],
    }
    if path is None:
        return merged
    import yaml  # local import — avoid hard dep at module load

    p = Path(path)
    if not p.exists():
        logger.warning("alert config not found: %s — using neutral defaults", p)
        return merged
    with p.open() as fh:
        data = yaml.safe_load(fh) or {}
    alerts = data.get("alerts", {}) if isinstance(data, dict) else {}
    if isinstance(alerts.get("confidence_thresholds"), dict):
        merged["confidence_thresholds"].update(alerts["confidence_thresholds"])
    if isinstance(alerts.get("frame_windows"), dict):
        merged["frame_windows"].update(alerts["frame_windows"])
    if "window_ratio" in alerts:
        merged["window_ratio"] = alerts["window_ratio"]
    if "cooldown_frames" in alerts:
        merged["cooldown_frames"] = alerts["cooldown_frames"]
    return merged


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------


class VideoProcessor:
    """Process video frames with object detection and configurable alert logic.

    Args:
        predictor: A :class:`DetectionPredictor` instance.
        alert_config: Alert configuration dictionary. Keys:

            - ``confidence_thresholds``: ``{class_name: float}``
            - ``frame_windows``: ``{class_name: int}``
              (classes without an entry trigger immediately)
            - ``window_ratio``: fraction of window that must be violations
            - ``cooldown_frames``: minimum frames between repeated alerts

            Uses sensible defaults when *None*.
    """

    def __init__(
        self,
        predictor: DetectionPredictor,
        alert_config: dict[str, Any] | None = None,
        enable_tracking: bool = False,
        tracker_config: dict[str, Any] | None = None,
        pose_predictor: Any | None = None,
        face_predictor: Any | None = None,
    ) -> None:
        self.predictor = predictor
        self.pose_predictor = pose_predictor
        self.face_predictor = face_predictor
        self.alert_config = alert_config or _NEUTRAL_ALERT_DEFAULTS

        # --- Tracking ---
        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        self._tracker: Any = None
        if enable_tracking:
            try:
                self._tracker = _sv_bridge.create_tracker(tracker_config)
                logger.info("ByteTrack tracking enabled.")
            except Exception as exc:
                logger.warning("Failed to create tracker: %s — tracking disabled.", exc)
                self.enable_tracking = False

        # --- Supervision annotators ---
        self._sv_annotators = _sv_bridge.build_annotators()

        # Pre-resolve alert config (constant for the lifetime of this processor)
        self._conf_thresholds = self.alert_config.get("confidence_thresholds", {})
        self._frame_windows = self.alert_config.get("frame_windows", {})
        self._window_ratio = self.alert_config.get("window_ratio", 0.8)
        self._cooldown_frames: int = self.alert_config.get("cooldown_frames", 90)

        # Reusable alert banner annotator
        self._alert_annotator = sv.LabelAnnotator(
            text_scale=0.7,
            text_thickness=2,
            text_padding=5,
            color=sv.Color(r=200, g=0, b=0),
            text_color=sv.Color(r=255, g=255, b=255),
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.INDEX,
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_stream(
        self,
        source: str,
        output_path: str | Path | None = None,
        show: bool = False,
        save_frames: bool = False,
        frame_dir: str | Path | None = None,
        skip_frames: int = 0,
        max_frames: int = 0,
        reconnect_delay: float = 5.0,
        max_reconnects: int = 10,
    ) -> dict[str, Any]:
        """Process a live stream (RTSP, HTTP, or device index).

        Supports RTSP URLs (``rtsp://...``), HTTP streams (``http://...``),
        and local camera devices (integer index like ``"0"``).

        Args:
            source: Stream URL or device index string.
            output_path: Path for annotated output video (optional).
            show: If *True*, display frames in a window (press ``q`` to quit).
            save_frames: If *True*, save annotated frames as images.
            frame_dir: Directory for saved frames.
            skip_frames: Process every N-th frame (0 = process all).
            max_frames: Stop after this many frames (0 = unlimited).
            reconnect_delay: Seconds to wait before reconnecting on failure.
            max_reconnects: Maximum reconnection attempts (0 = unlimited).

        Returns:
            Summary dict with processing statistics.
        """
        self._reset_state()

        # Resolve source (integer device or URL string)
        cap_source: Any = source
        if source.isdigit():
            cap_source = int(source)

        all_alerts: list[dict[str, Any]] = []
        class_counts: dict[str, int] = defaultdict(int)
        t_start = time.time()
        processed = 0
        total_detections = 0
        reconnects = 0
        stride = max(skip_frames + 1, 1)
        frame_idx = 0

        resolved_frame_dir: Path | None = None
        if save_frames:
            resolved_frame_dir = Path(frame_dir) if frame_dir else Path("stream_frames")
            resolved_frame_dir.mkdir(parents=True, exist_ok=True)

        writer = None

        while max_reconnects == 0 or reconnects <= max_reconnects:
            cap = cv2.VideoCapture(cap_source)
            if not cap.isOpened():
                reconnects += 1
                logger.warning(
                    "Failed to open stream '%s' (attempt %d/%s). "
                    "Retrying in %.1fs...",
                    source, reconnects,
                    max_reconnects or "inf", reconnect_delay,
                )
                time.sleep(reconnect_delay)
                continue

            fps_stream = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(
                "Stream connected: %s — %.1f FPS, %dx%d",
                source, fps_stream, width, height,
            )

            if output_path is not None and writer is None:
                out_p = Path(output_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                writer = _H264Writer(out_p, width, height, fps_stream)

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Stream read failed — attempting reconnect.")
                        break

                    frame_idx += 1
                    if frame_idx % stride != 0:
                        continue

                    annotated, detections, alerts = self.process_frame(
                        frame, frame_idx
                    )
                    processed += 1
                    total_detections += len(detections.get("labels", []))
                    for name in detections.get("class_names", []):
                        class_counts[name] += 1
                    all_alerts.extend(alerts)

                    if writer is not None:
                        writer.write_frame(annotated)
                    if save_frames and resolved_frame_dir is not None:
                        cv2.imwrite(
                            str(resolved_frame_dir / f"frame_{frame_idx:06d}.jpg"),
                            annotated,
                        )
                    if show:
                        cv2.imshow("Stream Detection", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("User interrupted at frame %d", frame_idx)
                            cap.release()
                            if writer is not None:
                                writer.close()
                            if show:
                                cv2.destroyAllWindows()
                            break

                    if 0 < max_frames <= processed:
                        logger.info("Reached max_frames=%d", max_frames)
                        cap.release()
                        break
                else:
                    # Stream read failed — reconnect
                    cap.release()
                    reconnects += 1
                    time.sleep(reconnect_delay)
                    continue

                break  # Normal exit (user quit or max_frames reached)

            except Exception as exc:
                logger.error("Stream processing error: %s", exc)
                cap.release()
                reconnects += 1
                time.sleep(reconnect_delay)

        if writer is not None:
            writer.close()
        if show:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        processing_fps = processed / elapsed if elapsed > 0 else 0.0
        return {
            "total_frames": frame_idx,
            "processed_frames": processed,
            "total_detections": total_detections,
            "alerts": all_alerts,
            "alert_count": len(all_alerts),
            "fps": round(processing_fps, 2),
            "class_counts": dict(class_counts),
            "elapsed_seconds": round(elapsed, 2),
            "reconnects": reconnects,
        }

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
        save_frames: bool = False,
        frame_dir: str | Path | None = None,
        skip_frames: int = 0,
    ) -> dict[str, Any]:
        """Process an entire video file.

        Args:
            video_path: Path to input video.
            output_path: Path for annotated output video (optional).
            show: If *True*, display frames in a window (press ``q`` to quit).
            save_frames: If *True*, save each annotated frame as an image.
            frame_dir: Directory for saved frames (defaults to
                ``<video_stem>_frames/``).
            skip_frames: Process every N-th frame (0 = process all).

        Returns:
            Summary dict with keys: ``total_frames``, ``processed_frames``,
            ``total_detections``, ``alerts``, ``fps``, ``class_counts``.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # --- Video metadata (needed for both paths) ---
        _cap = cv2.VideoCapture(str(video_path))
        if not _cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps_video = _cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()

        logger.info(
            "Video: %s — %d frames, %.1f FPS, %dx%d",
            video_path.name, total_frames, fps_video, width, height,
        )

        # Reset alert state
        self._reset_state()

        # Processing loop accumulators
        all_alerts: list[dict[str, Any]] = []
        class_counts: dict[str, int] = defaultdict(int)
        t_start = time.time()

        # Normalize frame_dir to Path for downstream methods
        resolved_frame_dir: Path | None = None
        if save_frames:
            if frame_dir is None:
                resolved_frame_dir = video_path.parent / f"{video_path.stem}_frames"
            else:
                resolved_frame_dir = Path(frame_dir)
            resolved_frame_dir.mkdir(parents=True, exist_ok=True)

        summary = self._process_video_sv(
            video_path, output_path, show, save_frames, resolved_frame_dir,
            skip_frames, fps_video, total_frames, width, height,
            all_alerts, class_counts, t_start,
        )

        logger.info(
            "Done: %d/%d frames, %d detections, %d alerts, %.1f FPS",
            summary["processed_frames"], summary["total_frames"],
            summary["total_detections"], summary["alert_count"],
            summary["fps"],
        )
        return summary

    # ------------------------------------------------------------------
    # Video I/O backends
    # ------------------------------------------------------------------

    def _process_video_sv(
        self,
        video_path: Path,
        output_path: str | Path | None,
        show: bool,
        save_frames: bool,
        frame_dir: Path | None,
        skip_frames: int,
        fps_video: float,
        total_frames: int,
        width: int,
        height: int,
        all_alerts: list[dict[str, Any]],
        class_counts: dict[str, int],
        t_start: float,
    ) -> dict[str, Any]:
        """Process video using supervision video I/O."""
        assert sv is not None  # narrowing for type checkers
        stride = max(skip_frames + 1, 1)
        frames_gen = sv.get_video_frames_generator(
            source_path=str(video_path), stride=stride,
        )

        # Prepare output path
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        processed = 0
        total_detections = 0
        frame_idx = 0

        writer = None
        try:
            if output_path is not None:
                writer = _H264Writer(output_path, width, height, fps_video)

            for frame in frames_gen:
                actual_idx = frame_idx * stride
                annotated, detections, alerts = self.process_frame(frame, actual_idx)
                processed += 1
                total_detections += len(detections.get("labels", []))

                for name in detections.get("class_names", []):
                    class_counts[name] += 1
                all_alerts.extend(alerts)

                if writer is not None:
                    writer.write_frame(annotated)
                if save_frames and frame_dir is not None:
                    cv2.imwrite(
                        str(frame_dir / f"frame_{actual_idx:06d}.jpg"), annotated
                    )
                if show:
                    cv2.imshow("Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User interrupted at frame %d", actual_idx)
                        break

                frame_idx += 1
        finally:
            if writer is not None:
                writer.close()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        processing_fps = processed / elapsed if elapsed > 0 else 0.0
        return {
            "total_frames": total_frames,
            "processed_frames": processed,
            "total_detections": total_detections,
            "alerts": all_alerts,
            "alert_count": len(all_alerts),
            "fps": round(processing_fps, 2),
            "class_counts": dict(class_counts),
            "elapsed_seconds": round(elapsed, 2),
        }

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
        """Process a single video frame.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Current frame index (used for alert tracking).

        Returns:
            Tuple of (annotated_frame, detections_dict, alert_list).
        """
        self._frame_count += 1
        detections = self.predictor.predict(frame)

        # Optional pose estimation on detected persons
        if self.pose_predictor is not None:
            self.pose_predictor.predict(frame)

        # Optional face recognition on violation detections
        face_results = None
        if self.face_predictor is not None:
            face_results = self.face_predictor.identify(frame, detections)

        alerts = self._check_alerts(detections, frame_idx, face_results=face_results)

        # --- Annotation via supervision ---
        sv_dets = _sv_bridge.to_sv_detections(detections)

        if self.enable_tracking and self._tracker is not None:
            sv_dets = _sv_bridge.update_tracker(self._tracker, sv_dets)

        annotated = _sv_bridge.annotate_frame(
            frame, sv_dets, self.predictor.class_names,
            self._sv_annotators,
            draw_traces=self.enable_tracking,
        )

        # Overlay alert banners
        annotated = self._draw_alerts(annotated, alerts)

        return annotated, detections, alerts

    # ------------------------------------------------------------------
    # Alert logic
    # ------------------------------------------------------------------

    def _check_alerts(
        self,
        detections: dict[str, np.ndarray],
        frame_idx: int,
        face_results: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Check detections against alert rules.

        Alert rules (from project spec):
        - **Fire/smoke**: trigger immediately if confidence exceeds threshold.
        - **PPE violations** (no_helmet, no_safety_shoes): require
          ``frame_windows[class]`` consecutive violation frames with at
          least ``window_ratio`` of them being violations.
        - **Fall detection**: multi-frame confirmation with a shorter window.

        Args:
            detections: Prediction dict from :meth:`DetectionPredictor.predict`.
            frame_idx: Current frame index.
            face_results: Optional face recognition results from
                :class:`FacePredictor.identify`.

        Returns:
            List of alert dicts with keys: ``type``, ``confidence``,
            ``message``, ``frame_idx``, and optionally ``identities``.
        """
        conf_thresholds = self._conf_thresholds
        frame_windows = self._frame_windows
        window_ratio = self._window_ratio
        cooldown = self._cooldown_frames

        alerts: list[dict[str, Any]] = []

        # Collect per-class identity info from face recognition results
        _face_identities: dict[str, list[str]] = {}
        if face_results is not None:
            for i, identity in enumerate(face_results.get("identities", [])):
                if identity is not None and identity != "unknown":
                    names = detections.get("class_names", [])
                    if i < len(names):
                        _face_identities.setdefault(names[i], []).append(identity)

        # Track which classes had a violation this frame
        violation_classes: dict[str, float] = {}
        for i, name in enumerate(detections.get("class_names", [])):
            score = float(detections["scores"][i])
            threshold = conf_thresholds.get(name)
            if threshold is not None and score >= threshold:
                # Keep the highest confidence for this class
                if name not in violation_classes or score > violation_classes[name]:
                    violation_classes[name] = score

        # Update rolling counters for all tracked classes
        for class_name in set(self._violation_history.keys()) | set(violation_classes.keys()):
            if class_name in violation_classes:
                self._violation_history[class_name].append(1)
            else:
                self._violation_history[class_name].append(0)

            window = frame_windows.get(class_name)

            if window is None:
                # Immediate trigger (e.g. fire, smoke)
                if class_name in violation_classes:
                    # Respect cooldown
                    last = self._last_alert_frame.get(class_name, -cooldown - 1)
                    if frame_idx - last >= cooldown:
                        alert_dict: dict[str, Any] = {
                            "type": class_name,
                            "confidence": violation_classes[class_name],
                            "message": (
                                f"ALERT: {class_name} detected "
                                f"(conf={violation_classes[class_name]:.2f})"
                            ),
                            "frame_idx": frame_idx,
                        }
                        if class_name in _face_identities:
                            alert_dict["identities"] = _face_identities[class_name]
                        alerts.append(alert_dict)
                        self._last_alert_frame[class_name] = frame_idx
            else:
                # Window-based confirmation
                history = self._violation_history[class_name]
                # Keep history bounded
                if len(history) > window * 2:
                    self._violation_history[class_name] = history[-window:]
                    history = self._violation_history[class_name]

                if len(history) >= window:
                    recent = history[-window:]
                    ratio = sum(recent) / window
                    if ratio >= window_ratio:
                        last = self._last_alert_frame.get(class_name, -cooldown - 1)
                        if frame_idx - last >= cooldown:
                            recent_peaks = self._peak_conf.get(class_name, [])
                            best_conf = violation_classes.get(
                                class_name,
                                recent_peaks[-1] if recent_peaks else 0.0,
                            )
                            alert_dict_w: dict[str, Any] = {
                                "type": class_name,
                                "confidence": best_conf,
                                "message": (
                                    f"ALERT: {class_name} sustained violation "
                                    f"({ratio:.0%} of last {window} frames, "
                                    f"conf={best_conf:.2f})"
                                ),
                                "frame_idx": frame_idx,
                            }
                            if class_name in _face_identities:
                                alert_dict_w["identities"] = _face_identities[class_name]
                            alerts.append(alert_dict_w)
                            self._last_alert_frame[class_name] = frame_idx
                            # Reset history after alert
                            self._violation_history[class_name] = []

        # Track peak confidence for window-based classes
        for cls_name, conf in violation_classes.items():
            self._peak_conf.setdefault(cls_name, []).append(conf)
            # Keep bounded
            if len(self._peak_conf[cls_name]) > 100:
                self._peak_conf[cls_name] = self._peak_conf[cls_name][-50:]

        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset alert tracking state for a new video."""
        self._frame_count: int = 0
        self._violation_history: dict[str, list[int]] = defaultdict(list)
        self._last_alert_frame: dict[str, int] = {}
        self._peak_conf: dict[str, list[float]] = {}
        if self.enable_tracking and self._tracker is not None:
            with contextlib.suppress(Exception):
                self._tracker = _sv_bridge.create_tracker(self.tracker_config)

    def _draw_alerts(
        self, image: np.ndarray, alerts: list[dict[str, Any]]
    ) -> np.ndarray:
        """Overlay alert banners on the top of the image.

        Args:
            image: Annotated BGR image.
            alerts: List of alert dicts.

        Returns:
            Image with alert banners drawn.
        """
        if not alerts:
            return image

        vis = image.copy()
        _, w = vis.shape[:2]

        n = len(alerts)
        xyxy = np.array(
            [[0, 30 + i * 30, w // 3, 30 + (i + 1) * 30] for i in range(n)],
            dtype=np.float32,
        )
        alert_dets = sv.Detections(xyxy=xyxy)
        labels = [a["message"] for a in alerts]

        vis = self._alert_annotator.annotate(scene=vis, detections=alert_dets, labels=labels)
        return vis
