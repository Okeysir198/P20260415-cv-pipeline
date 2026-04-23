"""Generate synthetic demo videos for each use case.

Creates 5 short videos (5-8 seconds each, 640x480, 30fps) with simple
animated shapes that visually represent each detection scenario.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.viz import (  # noqa: E402
    VizStyle,
    annotate_detections,
    annotate_polygons,
    classification_banner,
)

OUTPUT_DIR = Path(__file__).parent
FPS = 30
W, H = 640, 480


def _det_overlay_bgr(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    labels: list[str],
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Draw detection boxes+labels on a BGR frame using utils.viz helpers.

    Converts BGR→RGB at the boundary, runs ``annotate_detections``, then back
    to BGR. Colors are specified in RGB to match ``utils.viz`` convention.
    """
    if len(xyxy) == 0:
        return frame_bgr
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    dets = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
        class_id=np.zeros(len(xyxy), dtype=int),
    )
    r, g, b = color_rgb
    rgb = annotate_detections(
        rgb,
        detections=dets,
        labels=labels,
        color=sv.Color(r=r, g=g, b=b),
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _banner_bgr(
    frame_bgr: np.ndarray,
    text: str,
    bg_rgb: tuple[int, int, int] = (34, 34, 34),
    text_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Overlay a top banner on a BGR frame via utils.viz.classification_banner."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    style = VizStyle(banner_height=28, banner_text_scale=0.7)
    rgb = classification_banner(
        rgb,
        text=text,
        style=style,
        position="overlay_top",
        bg_color_rgb=bg_rgb,
        text_color_rgb=text_rgb,
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _polygon_zone_bgr(
    frame_bgr: np.ndarray,
    zone_pts: np.ndarray,
    label: str,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw a translucent restricted-zone polygon with an outline + label."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = color_rgb
    rgb = annotate_polygons(
        rgb,
        polygons=[zone_pts],
        labels=[label],
        color=sv.Color(r=r, g=g, b=b),
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_person(frame, cx, cy, scale=1.0, color=(0, 200, 0)):
    """Draw a simple stick figure person."""
    s = scale
    # Head
    cv2.circle(frame, (int(cx), int(cy - 60 * s)), int(15 * s), color, -1)
    # Body
    cv2.line(frame, (int(cx), int(cy - 45 * s)), (int(cx), int(cy + 20 * s)), color, max(2, int(3 * s)))
    # Arms
    cv2.line(frame, (int(cx - 30 * s), int(cy - 20 * s)), (int(cx + 30 * s), int(cy - 20 * s)), color, max(2, int(3 * s)))
    # Legs
    cv2.line(frame, (int(cx), int(cy + 20 * s)), (int(cx - 20 * s), int(cy + 60 * s)), color, max(2, int(3 * s)))
    cv2.line(frame, (int(cx), int(cy + 20 * s)), (int(cx + 20 * s), int(cy + 60 * s)), color, max(2, int(3 * s)))


def draw_helmet(frame, cx, cy, has_helmet=True):
    """Draw a person with or without a helmet."""
    draw_person(frame, cx, cy, color=(0, 200, 0) if has_helmet else (0, 0, 220))
    if has_helmet:
        # Helmet (yellow arc on top of head)
        cv2.ellipse(frame, (int(cx), int(cy - 65)), (18, 12), 0, 180, 360, (0, 220, 220), -1)
        cv2.rectangle(frame, (int(cx - 18), int(cy - 70)), (int(cx + 18), int(cy - 65)), (0, 220, 220), -1)


def create_fire_smoke_video():
    """Fire/smoke scene: flickering flames and rising smoke particles."""
    path = str(OUTPUT_DIR / "demo_fire_smoke.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 6  # 6 seconds

    np.random.seed(42)
    # Pre-generate smoke particles
    smoke_particles = []
    for _ in range(80):
        smoke_particles.append({
            'x': np.random.randint(200, 450),
            'y': np.random.randint(100, 400),
            'vx': np.random.uniform(-0.5, 0.5),
            'vy': np.random.uniform(-2.5, -0.8),
            'r': np.random.randint(8, 25),
            'life': np.random.randint(0, n_frames),
        })

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (30, 25, 20)  # Dark background

        # Ground
        cv2.rectangle(frame, (0, 400), (W, H), (40, 50, 40), -1)

        # Building silhouette
        cv2.rectangle(frame, (180, 150), (460, 400), (60, 55, 50), -1)
        cv2.rectangle(frame, (220, 180), (280, 250), (50, 45, 40), -1)  # Window
        cv2.rectangle(frame, (350, 180), (410, 250), (50, 45, 40), -1)  # Window

        # Fire at base of building (flickering polygons)
        for _ in range(15):
            fx = np.random.randint(200, 440)
            fy = np.random.randint(300, 400)
            fh = np.random.randint(30, 80 + int(20 * math.sin(i * 0.3)))
            fw = np.random.randint(10, 30)
            pts = np.array([
                [fx, fy],
                [fx - fw, fy],
                [fx - fw // 2 + np.random.randint(-5, 5), fy - fh],
            ], np.int32)
            r = np.random.randint(200, 255)
            g = np.random.randint(80, 180)
            cv2.fillPoly(frame, [pts], (0, g, r))

        # Orange glow
        overlay = frame.copy()
        cv2.rectangle(overlay, (180, 280), (460, 400), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.2 + 0.05 * math.sin(i * 0.5), frame, 0.8 - 0.05 * math.sin(i * 0.5), 0, frame)

        # Smoke particles (gray circles rising)
        for p in smoke_particles:
            age = (i - p['life']) % (n_frames // 2)
            if age < 0:
                continue
            px = int(p['x'] + p['vx'] * age + 5 * math.sin(age * 0.1))
            py = int(p['y'] + p['vy'] * age)
            alpha = max(0, 1 - age / 60)
            if 0 < py < H and 0 < px < W and alpha > 0:
                gray = int(120 + 60 * alpha)
                r = int(p['r'] * (1 + age * 0.02))
                cv2.circle(frame, (px, py), r, (gray, gray, gray + 10), -1)

        # Title banner (annotation overlay) — via utils.viz
        frame = _banner_bgr(frame, "FIRE/SMOKE DETECTION DEMO")
        # Frame counter — pure UI chrome, not a detection annotation (KEEP cv2)
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        out.write(frame)
    out.release()
    print(f"  Created: {path}")


def create_construction_ppe_video():
    """Workers with and without helmets walking across a site."""
    path = str(OUTPUT_DIR / "demo_construction_ppe.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 7  # 7 seconds

    workers = [
        {'x_start': -50, 'speed': 2.0, 'y': 300, 'helmet': True},
        {'x_start': 700, 'speed': -1.5, 'y': 320, 'helmet': False},
        {'x_start': -100, 'speed': 1.2, 'y': 340, 'helmet': True},
        {'x_start': 750, 'speed': -1.8, 'y': 290, 'helmet': False},
        {'x_start': 100, 'speed': 0.8, 'y': 310, 'helmet': True},
    ]

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Construction site background
        frame[:] = (180, 200, 210)  # Light sky
        cv2.rectangle(frame, (0, 360), (W, H), (120, 140, 150), -1)  # Ground
        # Scaffolding
        for sx in range(50, W, 120):
            cv2.line(frame, (sx, 100), (sx, 360), (100, 100, 100), 3)
            cv2.line(frame, (sx, 200), (sx + 120, 200), (100, 100, 100), 3)
            cv2.line(frame, (sx, 280), (sx + 120, 280), (100, 100, 100), 3)

        # Warning stripes at bottom
        for sx in range(0, W, 40):
            cv2.rectangle(frame, (sx, H - 30), (sx + 20, H), (0, 200, 200), -1)

        # Draw scene stick-figures first, then stack detection annotations.
        for w in workers:
            wx = int(w['x_start'] + w['speed'] * i)
            if -50 < wx < W + 50:
                draw_helmet(frame, wx, w['y'], w['helmet'])

        # Detection overlays (bbox + label around each worker) — via utils.viz
        ok_boxes, ok_labels = [], []
        bad_boxes, bad_labels = [], []
        for w in workers:
            wx = int(w['x_start'] + w['speed'] * i)
            if not (-50 < wx < W + 50):
                continue
            box = [wx - 30, w['y'] - 80, wx + 30, w['y'] + 65]
            if w['helmet']:
                ok_boxes.append(box)
                ok_labels.append("HELMET")
            else:
                bad_boxes.append(box)
                bad_labels.append("NO HELMET")
        if ok_boxes:
            frame = _det_overlay_bgr(frame, np.array(ok_boxes), ok_labels, (0, 180, 0))
        if bad_boxes:
            frame = _det_overlay_bgr(frame, np.array(bad_boxes), bad_labels, (220, 0, 0))

        frame = _banner_bgr(frame, "CONSTRUCTION PPE DETECTION DEMO",
                            bg_rgb=(230, 230, 230), text_rgb=(40, 40, 40))
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out.write(frame)
    out.release()
    print(f"  Created: {path}")


def create_fall_detection_video():
    """Person walking then falling down."""
    path = str(OUTPUT_DIR / "demo_fall_detection.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 7  # 7 seconds

    fall_start = int(n_frames * 0.45)
    fall_duration = 20

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (220, 215, 210)  # Indoor background
        cv2.rectangle(frame, (0, 380), (W, H), (160, 150, 140), -1)  # Floor
        # Wall/door
        cv2.rectangle(frame, (500, 100), (560, 380), (140, 130, 120), -1)

        cx = 150 + i * 1.5
        cy_base = 310

        if i < fall_start:
            # Walking normally with slight bob
            bob = 5 * math.sin(i * 0.5)
            draw_person(frame, cx, cy_base + bob, scale=1.2, color=(50, 150, 50))
            # Detection-style label around the walking person
            frame = _det_overlay_bgr(
                frame,
                np.array([[cx - 30, cy_base + bob - 105, cx + 30, cy_base + bob + 65]]),
                ["NORMAL"],
                (50, 150, 50),
            )
        elif i < fall_start + fall_duration:
            # Falling animation — tilting body is SCENE figure, KEEP cv2 primitives
            progress = (i - fall_start) / fall_duration
            angle = progress * 80  # degrees
            fall_cx = cx + progress * 30
            fall_cy = cy_base + progress * 50
            rad = math.radians(angle)
            body_len = 65 * 1.2
            head_x = int(fall_cx - math.sin(rad) * body_len * 0.6)
            head_y = int(fall_cy - math.cos(rad) * body_len * 0.6)
            foot_x = int(fall_cx + math.sin(rad) * body_len * 0.4)
            foot_y = int(fall_cy + math.cos(rad) * body_len * 0.4)
            cv2.circle(frame, (head_x, head_y - 15), 18, (0, 0, 220), -1)
            cv2.line(frame, (head_x, head_y), (foot_x, foot_y), (0, 0, 220), 4)
            # Detection-style label overlay
            frame = _det_overlay_bgr(
                frame,
                np.array([[fall_cx - 45, fall_cy - 95, fall_cx + 45, fall_cy + 30]]),
                ["FALLING!"],
                (255, 0, 0),
            )
        else:
            # Fallen on ground — body shape is scene, KEEP cv2 primitives
            gx = cx + 30
            gy = 370
            cv2.ellipse(frame, (int(gx), int(gy)), (50, 15), 0, 0, 360, (0, 0, 200), -1)
            cv2.circle(frame, (int(gx - 40), int(gy - 5)), 15, (0, 0, 200), -1)
            # Blinking detection alert (box + label) — MIGRATED to utils.viz
            if (i // 8) % 2 == 0:
                frame = _det_overlay_bgr(
                    frame,
                    np.array([[gx - 70, gy - 30, gx + 70, gy + 20]]),
                    ["!! FALL DETECTED !!"],
                    (255, 0, 0),
                )

        frame = _banner_bgr(frame, "FALL DETECTION DEMO",
                            bg_rgb=(230, 230, 230), text_rgb=(40, 40, 40))
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out.write(frame)
    out.release()
    print(f"  Created: {path}")


def create_phone_usage_video():
    """Person standing and looking at phone."""
    path = str(OUTPUT_DIR / "demo_phone_usage.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 6  # 6 seconds

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (200, 195, 190)  # Office background
        cv2.rectangle(frame, (0, 380), (W, H), (150, 140, 130), -1)  # Floor
        # Desk
        cv2.rectangle(frame, (50, 280), (250, 300), (100, 80, 60), -1)
        cv2.rectangle(frame, (400, 280), (600, 300), (100, 80, 60), -1)

        # Person 1: not using phone (working at desk) — scene figure
        draw_person(frame, 150, 280, color=(50, 150, 50))
        # Detection label for "WORKING" (green, classification-style)
        frame = _det_overlay_bgr(
            frame, np.array([[110, 180, 200, 290]]), ["WORKING"], (50, 150, 50)
        )

        # Person 2: using phone (looking down with phone rectangle) — scene figure
        px2 = 500
        py2 = 260
        draw_person(frame, px2, py2, color=(0, 0, 220))
        # Phone in hand (glowing rectangle) — pure scene prop, KEEP cv2
        phone_x = int(px2 + 15 + 3 * math.sin(i * 0.15))
        phone_y = int(py2 - 15 + 2 * math.sin(i * 0.1))
        cv2.rectangle(frame, (phone_x, phone_y), (phone_x + 12, phone_y + 20), (255, 255, 200), -1)
        cv2.rectangle(frame, (phone_x, phone_y), (phone_x + 12, phone_y + 20), (200, 200, 150), 1)
        # Blinking detection alert (box + label) — MIGRATED
        if (i // 15) % 2 == 0:
            frame = _det_overlay_bgr(
                frame,
                np.array([[px2 - 50, py2 - 80, px2 + 50, py2 + 70]]),
                ["PHONE DETECTED!"],
                (255, 0, 0),
            )

        # Person 3: walking by — scene figure
        p3x = int(320 + 50 * math.sin(i * 0.04))
        draw_person(frame, p3x, 320, scale=0.8, color=(50, 150, 50))

        frame = _banner_bgr(frame, "PHONE USAGE DETECTION DEMO",
                            bg_rgb=(230, 230, 230), text_rgb=(40, 40, 40))
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out.write(frame)
    out.release()
    print(f"  Created: {path}")


def create_zone_intrusion_video():
    """People walking, some entering a restricted zone."""
    path = str(OUTPUT_DIR / "demo_zone_intrusion.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 8  # 8 seconds

    # Restricted zone polygon
    zone_pts = np.array([[250, 180], [450, 180], [480, 380], [220, 380]], np.int32)

    people = [
        {'x_start': -30, 'speed': 1.8, 'y': 300},   # Walks through zone
        {'x_start': 50, 'speed': 0.5, 'y': 350},     # Walks along bottom edge
        {'x_start': 600, 'speed': -1.2, 'y': 280},   # Walks through zone from right
        {'x_start': 500, 'speed': -0.3, 'y': 150},   # Walks above zone
    ]

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (200, 200, 200)  # Outdoor area
        cv2.rectangle(frame, (0, 390), (W, H), (140, 150, 140), -1)  # Ground

        # Restricted zone (translucent polygon + outline + label) — MIGRATED
        frame = _polygon_zone_bgr(frame, zone_pts, "RESTRICTED ZONE", color_rgb=(255, 0, 0))

        intrusion_boxes: list[list[int]] = []
        intrusion_labels: list[str] = []
        blink_on = (i // 10) % 2 == 0
        for p in people:
            px = int(p['x_start'] + p['speed'] * i)
            py = p['y']
            if -50 < px < W + 50:
                inside = cv2.pointPolygonTest(zone_pts, (float(px), float(py)), False) >= 0
                color = (0, 0, 220) if inside else (50, 180, 50)
                draw_person(frame, px, py, scale=1.0, color=color)
                if inside:
                    intrusion_boxes.append([px - 30, py - 85, px + 30, py + 65])
                    intrusion_labels.append("INTRUSION!" if blink_on else "")
        if intrusion_boxes:
            frame = _det_overlay_bgr(
                frame, np.array(intrusion_boxes), intrusion_labels, (255, 0, 0)
            )

        frame = _banner_bgr(frame, "ZONE INTRUSION DETECTION DEMO",
                            bg_rgb=(230, 230, 230), text_rgb=(40, 40, 40))
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out.write(frame)
    out.release()
    print(f"  Created: {path}")


def create_general_demo_video():
    """General-purpose demo with multiple people walking -- usable across all tabs."""
    path = str(OUTPUT_DIR / "demo_general.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    n_frames = FPS * 8  # 8 seconds

    people = [
        {'x_start': -40, 'speed': 1.5, 'y': 300, 'scale': 1.1},
        {'x_start': 700, 'speed': -2.0, 'y': 320, 'scale': 1.0},
        {'x_start': 100, 'speed': 0.8, 'y': 280, 'scale': 0.9},
        {'x_start': 500, 'speed': -0.6, 'y': 350, 'scale': 1.2},
        {'x_start': -80, 'speed': 1.0, 'y': 260, 'scale': 0.8},
        {'x_start': 300, 'speed': 0.3, 'y': 340, 'scale': 1.0},
    ]

    for i in range(n_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Outdoor scene
        frame[:200] = (210, 190, 160)  # Sky
        frame[200:] = (170, 180, 170)  # Ground area
        cv2.rectangle(frame, (0, 380), (W, H), (130, 140, 130), -1)  # Pavement
        # Some structures
        cv2.rectangle(frame, (20, 100), (120, 380), (150, 145, 140), -1)
        cv2.rectangle(frame, (550, 80), (630, 380), (145, 140, 135), -1)

        for p in people:
            px = int(p['x_start'] + p['speed'] * i)
            if -50 < px < W + 50:
                bob = 4 * math.sin(i * 0.4 + p['x_start'])
                draw_person(frame, px, int(p['y'] + bob), scale=p['scale'], color=(60, 160, 60))

        frame = _banner_bgr(frame, "GENERAL DEMO (MULTI-PURPOSE)",
                            bg_rgb=(230, 230, 230), text_rgb=(40, 40, 40))
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        out.write(frame)
    out.release()
    print(f"  Created: {path}")


if __name__ == "__main__":
    print("Generating demo videos...")
    create_fire_smoke_video()
    create_construction_ppe_video()
    create_fall_detection_video()
    create_phone_usage_video()
    create_zone_intrusion_video()
    create_general_demo_video()
    print("\nDone! All videos saved to:", OUTPUT_DIR)
