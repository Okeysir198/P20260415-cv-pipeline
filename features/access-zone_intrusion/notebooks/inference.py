#!/usr/bin/env python3
"""Run inference with Zone Intrusion Detection model.

Usage:
    python features/access-zone_intrusion/experiments/inference.py --image path/to/image.jpg
    python features/access-zone_intrusion/experiments/inference.py --video path/to/video.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p10_inference.predictor import DetectionPredictor
from utils.config import load_config

DEFAULT_MODEL = "runs/zone_intrusion/best.pt"
DEFAULT_DATA_CONFIG = "features/access-zone_intrusion/configs/05_data.yaml"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Zone Intrusion Detection inference")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path (.pt or .onnx)")
    parser.add_argument("--config", default=DEFAULT_DATA_CONFIG, help="Data config path")
    parser.add_argument("--image", type=str, help="Image path for single-image inference")
    parser.add_argument("--video", type=str, help="Video path for video inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")
    args = parser.parse_args()

    data_config = load_config(args.config)
    predictor = DetectionPredictor(model_path=args.model, data_config=data_config, conf_threshold=args.conf)

    if args.image:
        import cv2

        image = cv2.imread(args.image)
        results = predictor.predict(image)
        print(f"Detections: {len(results.get('boxes', []))}")
        if args.save_dir:
            vis = predictor.visualize(image, results)
            save_path = Path(args.save_dir) / Path(args.image).name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            print(f"Saved to {save_path}")
    elif args.video:
        from core.p10_inference.video_inference import VideoProcessor

        processor = VideoProcessor(predictor=predictor)
        processor.process_video(args.video, save_dir=args.save_dir)
    else:
        parser.error("Provide --image or --video")


if __name__ == "__main__":
    main()
