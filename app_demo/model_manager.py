"""Model manager with lazy loading and caching for the demo app."""

import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.p06_models.face_registry import build_face_detector, build_face_embedder
from core.p06_models.pose_registry import build_pose_model
from core.p10_inference.face_gallery import FaceGallery
from core.p10_inference.face_predictor import FacePredictor
from core.p10_inference.pose_predictor import PosePredictor
from core.p10_inference.predictor import DetectionPredictor
from utils.config import load_config

_APP_DEMO_ROOT = Path(__file__).resolve().parent


def load_coco_data_config(config: dict, normalize: bool = True) -> dict:
    """Load COCO data config from app_demo/config folder.

    Args:
        config: Demo config dict containing ``coco_names_config`` key.
        normalize: If True, use ImageNet mean/std (for HF models).
            If False, use 0/1 mean/std (for YOLOX raw pixel input).
    """
    coco_config_path = config.get("coco_names_config", "config/coco_names.yaml")
    cfg = load_config(str(_APP_DEMO_ROOT / coco_config_path))

    if not normalize:
        cfg["mean"] = [0.0, 0.0, 0.0]
        cfg["std"] = [1 / 255, 1 / 255, 1 / 255]

    return cfg


class ModelManager:
    """Lazy model loading with caching for the demo app.

    Manages detection, pose, and face recognition models. Models are loaded
    on first access and cached for reuse. All model definitions come from
    config.yaml — no hardcoded model specs.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._cache: dict[str, Any] = {}
        self._project_root = Path(__file__).resolve().parent.parent

    def _coco_data_config(self, normalize: bool = True) -> dict:
        """Load COCO data config from app_demo/config folder.

        Args:
            normalize: If True, use ImageNet mean/std (for HF models).
                If False, use 0/1 mean/std (for YOLOX raw pixel input).
        """
        return load_coco_data_config(self._config, normalize=normalize)

    def _coco_models(self) -> dict[str, dict]:
        """Get COCO pretrained model specs from config."""
        return self._config.get("models", {}).get("coco_pretrained", {})

    def get_coco_predictor(
        self, conf_threshold: float = 0.5, model_name: str = "YOLOX-M"
    ) -> DetectionPredictor:
        """Load a COCO pretrained detection model.

        Args:
            conf_threshold: Detection confidence threshold.
            model_name: Key from config models.coco_pretrained (e.g. "YOLOX-M").
        """
        key = f"coco_{model_name}"
        if key not in self._cache:
            coco_models = self._coco_models()
            if model_name not in coco_models:
                raise ValueError(
                    f"Unknown COCO model '{model_name}'. "
                    f"Available: {list(coco_models)}"
                )
            spec = coco_models[model_name]
            normalize = spec.get("normalize", True)

            if "model_path" in spec:
                # Direct checkpoint (YOLOX-M)
                model_path = self._project_root / spec["model_path"]
            else:
                # HF model — build and cache as checkpoint
                model_path = self._project_root / spec["cache_path"]
                if not model_path.exists():
                    from core.p06_models import build_model
                    import torch

                    model = build_model(spec["hf_config"])
                    model.eval()
                    torch.save(
                        {"config": spec["hf_config"], "model": model.state_dict()},
                        str(model_path),
                    )
                    logger.info("Saved %s COCO checkpoint to %s", model_name, model_path)

            self._cache[key] = DetectionPredictor(
                model_path=str(model_path),
                data_config=self._coco_data_config(normalize=normalize),
                conf_threshold=conf_threshold,
            )
        predictor = self._cache[key]
        predictor.conf_threshold = conf_threshold
        return predictor

    def get_use_case_predictor(
        self, use_case: str, conf_threshold: float = 0.5
    ) -> tuple[DetectionPredictor, str]:
        """Load fine-tuned model for use case, or fall back to COCO.

        Searches model_paths from config in order; first existing file wins.

        Returns:
            Tuple of (predictor, model_type) where model_type is
            "fine-tuned" or "coco-pretrained".
        """
        fine_tuned_cfg = self._config.get("models", {}).get("fine_tuned", {})
        uc_model_cfg = fine_tuned_cfg.get(use_case, {})

        # Search configured paths in order
        model_path = None
        for candidate in uc_model_cfg.get("model_paths", []):
            full = self._project_root / candidate
            if full.exists():
                model_path = full
                break

        if model_path is not None:
            key = f"finetuned_{use_case}"
            if key not in self._cache:
                uc_config = self._config["use_cases"].get(use_case, {})
                data_config_path = uc_config.get("data_config")
                if data_config_path:
                    data_config = load_config(str(self._project_root / data_config_path))
                else:
                    data_config = self._coco_data_config()
                self._cache[key] = DetectionPredictor(
                    model_path=str(model_path),
                    data_config=data_config,
                    conf_threshold=conf_threshold,
                )
            predictor = self._cache[key]
            predictor.conf_threshold = conf_threshold
            return predictor, "fine-tuned"

        return self.get_coco_predictor(conf_threshold), "coco-pretrained"

    def get_feature_inference_config(self, name: str) -> dict:
        """Load ``features/<name>/configs/10_inference.yaml`` if present.

        Returns an empty dict when the file is missing so callers can safely
        chain ``.get(...)``. This is the authoritative source for per-feature
        alerts, tracker settings, and sample paths — tabs should call this
        instead of reading the top-level ``config.yaml → alerts`` block.
        """
        key = f"infer_cfg_{name}"
        if key in self._cache:
            return self._cache[key]
        path = self._project_root / "features" / name / "configs" / "10_inference.yaml"
        if not path.exists():
            self._cache[key] = {}
            return {}
        cfg = load_config(str(path))
        self._cache[key] = cfg
        return cfg

    def get_feature_alert_config(self, name: str) -> dict:
        """Per-feature alert config — falls back to neutral defaults."""
        from core.p10_inference.video_inference import _NEUTRAL_ALERT_DEFAULTS

        cfg = self.get_feature_inference_config(name)
        alerts = cfg.get("alerts") if isinstance(cfg, dict) else None
        if not isinstance(alerts, dict):
            return dict(_NEUTRAL_ALERT_DEFAULTS)
        merged = dict(_NEUTRAL_ALERT_DEFAULTS)
        for k, v in alerts.items():
            merged[k] = v
        return merged

    def get_pose_predictor(
        self,
        pose_config_path: str,
        det_predictor: Optional[DetectionPredictor] = None,
        person_class_ids: Optional[list[int]] = None,
    ) -> Optional[PosePredictor]:
        """Load pose predictor (RTMPose or MediaPipe).

        Returns None if pose model weights not found.
        """
        key = f"pose_{pose_config_path}"
        if key not in self._cache:
            config_path = self._project_root / pose_config_path
            if not config_path.exists():
                logger.warning("Pose config not found: %s", config_path)
                return None
            pose_config = load_config(str(config_path))

            model_path = pose_config.get("pose_model", {}).get("model_path", "")
            if model_path and not (self._project_root / model_path).exists():
                if "mediapipe" not in pose_config.get("pose_model", {}).get("arch", ""):
                    logger.warning("Pose model weights not found: %s", model_path)
                    return None

            pose_model = build_pose_model(pose_config)

            if det_predictor is None:
                det_predictor = self.get_coco_predictor()

            self._cache[key] = PosePredictor(
                detector=det_predictor,
                pose_model=pose_model,
                person_class_ids=person_class_ids or [0],
            )
        return self._cache[key]

    def get_face_predictor(
        self, violation_class_ids: Optional[list[int]] = None
    ) -> Optional[FacePredictor]:
        """Load face recognition predictor (SCRFD + MobileFaceNet + Gallery).

        Returns None if ONNX models not found.
        """
        key = "face_predictor"
        if key not in self._cache:
            face_cfg = self._config.get("face", {})
            face_config_path = face_cfg.get("config", "features/access-face_recognition/configs/face.yaml")
            config_path = self._project_root / face_config_path

            if not config_path.exists():
                logger.warning("Face config not found: %s", config_path)
                return None

            face_config = load_config(str(config_path))

            det_path = self._project_root / face_config["face_detector"]["model_path"]
            emb_path = self._project_root / face_config["face_embedder"]["model_path"]
            if not det_path.exists() or not emb_path.exists():
                logger.warning("Face ONNX models not found: %s, %s", det_path, emb_path)
                return None

            face_config["face_detector"]["model_path"] = str(det_path)
            face_config["face_embedder"]["model_path"] = str(emb_path)
            detector = build_face_detector(face_config)
            embedder = build_face_embedder(face_config)

            gallery_path = self._project_root / face_cfg.get(
                "gallery_path",
                face_config.get("gallery", {}).get("path", "data/face_gallery/demo.npz"),
            )
            gallery_path.parent.mkdir(parents=True, exist_ok=True)
            gallery = FaceGallery(
                gallery_path=str(gallery_path),
                similarity_threshold=face_config.get("gallery", {}).get("similarity_threshold", 0.4),
            )

            v_ids = violation_class_ids or face_config.get("inference", {}).get("violation_class_ids", [2])

            self._cache[key] = FacePredictor(
                face_detector=detector,
                face_embedder=embedder,
                gallery=gallery,
                violation_class_ids=v_ids,
                expand_ratio=face_config.get("inference", {}).get("expand_ratio", 1.5),
                face_conf_threshold=face_config.get("inference", {}).get("face_conf_threshold", 0.5),
            )
        return self._cache[key]

    def get_face_gallery(self) -> Optional[FaceGallery]:
        """Get the face gallery (loading face predictor if needed)."""
        predictor = self.get_face_predictor()
        if predictor is None:
            return None
        return predictor.gallery

    def warmup(self) -> None:
        """Preload all available models to GPU at startup."""
        logger.info("Warming up models...")

        # 1. All COCO pretrained detectors
        for name, spec in self._coco_models().items():
            # Skip HF models whose cache doesn't exist yet (will download on first use)
            if "model_path" in spec:
                path = self._project_root / spec["model_path"]
                if not path.exists():
                    logger.warning("  Skipping %s — weights not found at %s", name, path)
                    continue
            logger.info("  Loading COCO pretrained: %s...", name)
            self.get_coco_predictor(conf_threshold=0.25, model_name=name)

        # 2. Fine-tuned models
        for use_case in self.discover_fine_tuned():
            logger.info("  Loading fine-tuned model: %s", use_case)
            self.get_use_case_predictor(use_case, conf_threshold=0.25)

        # 3. Face recognition (SCRFD + MobileFaceNet ONNX)
        logger.info("  Loading face recognition models...")
        face_pred = self.get_face_predictor()
        if face_pred is not None:
            logger.info("  Face recognition ready")
        else:
            logger.warning("  Face ONNX models not found — skipping")

        logger.info("Warmup complete. %d models cached.", len(self._cache))

    def discover_fine_tuned(self) -> dict[str, Path]:
        """Find fine-tuned models by checking config-listed paths.

        Searches ``models.fine_tuned.<use_case>.model_paths`` in order;
        first existing file per use case wins.

        Returns:
            Dict mapping use-case name to model path.
        """
        fine_tuned_cfg = self._config.get("models", {}).get("fine_tuned", {})
        found: dict[str, Path] = {}
        for use_case, uc_cfg in fine_tuned_cfg.items():
            for candidate in uc_cfg.get("model_paths", []):
                full = self._project_root / candidate
                if full.exists():
                    found[use_case] = full
                    break
        return found

    def list_available_models(self) -> list[str]:
        """List all available model choices for the detection dropdown."""
        choices = []
        for name in self._coco_models():
            choices.append(f"COCO-{name} (pretrained)")
        for name in self.discover_fine_tuned():
            choices.append(f"{name} (fine-tuned)")
        return choices

    def get_predictor_by_choice(
        self, choice: str, conf_threshold: float = 0.5
    ) -> tuple[DetectionPredictor, str]:
        """Get predictor from a dropdown choice string."""
        if "pretrained" in choice.lower():
            # Extract model name: "COCO-YOLOX-M (pretrained)" -> "YOLOX-M"
            model_name = choice.replace("COCO-", "").split(" (")[0].strip()
            return self.get_coco_predictor(conf_threshold, model_name), f"coco-pretrained ({model_name})"
        name = choice.split(" (")[0].strip()
        return self.get_use_case_predictor(name, conf_threshold)
