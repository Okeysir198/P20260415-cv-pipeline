"""Model loader component for demo tabs.

Provides a generic model loading interface supporting both file browser uploads
(.pt/.onnx) and pre-configured model dropdowns. Uses DetectionPredictor from
core/p10_inference/predictor.py for all model types.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p10_inference.predictor import DetectionPredictor
from app_demo.model_manager import load_coco_data_config


class ModelLoader:
    """Generic model loading component for demo tabs.

    Supports two loading methods:
    1. File browser upload — user selects .pt or .onnx file
    2. Pre-configured dropdown — models defined in config.yaml

    All models are loaded via DetectionPredictor which handles both
    PyTorch and ONNX backends automatically.
    """

    def __init__(
        self,
        config: Dict,
        predefined_models: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the model loader.

        Args:
            config: Demo config dict containing model specifications.
            predefined_models: Optional dict mapping display names to model paths.
                If None, uses COCO pretrained models from config.
        """
        self._config = config
        self._project_root = Path(__file__).resolve().parent.parent.parent

        if predefined_models is not None:
            self._predefined_models = predefined_models
        else:
            # Extract COCO pretrained models from config
            coco_models = config.get("models", {}).get("coco_pretrained", {})
            self._predefined_models = {
                name: spec.get("model_path", spec.get("cache_path", ""))
                for name, spec in coco_models.items()
            }

        self._cache: Dict[str, DetectionPredictor] = {}

    def list_predefined_models(self) -> list[str]:
        """Get list of available predefined model names.

        Returns:
            List of display names for predefined models.
        """
        return list(self._predefined_models.keys())

    def build_ui(
        self,
        label: str = "Model",
        show_file_upload: bool = True,
        show_dropdown: bool = True,
    ) -> Tuple:
        """Build the Gradio UI components for model loading.

        Args:
            label: Label text for model selection components.
            show_file_upload: If True, show file browser component.
            show_dropdown: If True, show predefined model dropdown.

        Returns:
            Tuple of (file_component, dropdown_component, use_predefined_checkbox).
            Components not shown are None.
        """
        predefined_choices = self.list_predefined_models()
        default_predefined = predefined_choices[0] if predefined_choices else ""

        file_comp = None
        dropdown_comp = None
        checkbox_comp = None

        if show_file_upload and show_dropdown:
            # Both modes with radio toggle
            with gr.Row():
                use_predefined = gr.Radio(
                    choices=["Predefined", "Upload File"],
                    value="Predefined",
                    label="Model Source",
                    interactive=True,
                )
            file_comp = gr.File(
                label=f"{label} (Upload .pt/.onnx)",
                file_types=[".pt", ".pth", ".onnx"],
                visible=False,
                interactive=True,
            )
            dropdown_comp = gr.Dropdown(
                choices=predefined_choices,
                value=default_predefined,
                label=f"{label} (Predefined)",
                visible=True,
                interactive=True,
            )
            # Toggle visibility based on radio choice
            use_predefined.change(
                fn=lambda choice: (
                    gr.update(visible=(choice == "Upload File")),
                    gr.update(visible=(choice == "Predefined")),
                ),
                inputs=[use_predefined],
                outputs=[file_comp, dropdown_comp],
            )
            checkbox_comp = use_predefined

        elif show_file_upload:
            file_comp = gr.File(
                label=label,
                file_types=[".pt", ".pth", ".onnx"],
                interactive=True,
            )

        elif show_dropdown:
            dropdown_comp = gr.Dropdown(
                choices=predefined_choices,
                value=default_predefined,
                label=label,
                interactive=True,
            )

        return file_comp, dropdown_comp, checkbox_comp

    def load_from_file(
        self,
        file_path: Optional[str],
        conf_threshold: float = 0.25,
        data_config: Optional[Dict] = None,
    ) -> Optional[DetectionPredictor]:
        """Load model from uploaded file path.

        Args:
            file_path: Path to .pt or .onnx file.
            conf_threshold: Detection confidence threshold.
            data_config: Optional data config dict (class names, normalization).
                If None, uses default COCO config.

        Returns:
            DetectionPredictor instance, or None if file_path is invalid.
        """
        if not file_path:
            return None

        # Resolve relative paths from project root
        path = Path(file_path)
        if not path.is_absolute():
            path = self._project_root / file_path

        # Use default COCO data config if not provided
        if data_config is None:
            data_config = self._default_data_config()

        cache_key = f"file_{path}"
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = DetectionPredictor(
                    model_path=str(path),
                    data_config=data_config,
                    conf_threshold=conf_threshold,
                )
                logger.info("Loaded model from file: %s", path)
            except Exception as exc:
                logger.error("Failed to load model from %s: %s", path, exc)
                return None

        predictor = self._cache[cache_key]
        predictor.conf_threshold = conf_threshold
        return predictor

    def load_predefined(
        self,
        model_name: str,
        conf_threshold: float = 0.25,
    ) -> Optional[DetectionPredictor]:
        """Load predefined model by name.

        Args:
            model_name: Display name from predefined_models dict.
            conf_threshold: Detection confidence threshold.

        Returns:
            DetectionPredictor instance, or None if model_name is invalid.
        """
        if model_name not in self._predefined_models:
            logger.error("Unknown predefined model: %s", model_name)
            return None

        model_path = self._predefined_models[model_name]

        # Resolve relative paths from project root
        path = Path(model_path)
        if not path.is_absolute():
            path = self._project_root / model_path

        cache_key = f"predef_{model_name}"
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = DetectionPredictor(
                    model_path=str(path),
                    data_config=self._default_data_config(),
                    conf_threshold=conf_threshold,
                )
                logger.info("Loaded predefined model: %s", model_name)
            except Exception as exc:
                logger.error("Failed to load predefined model %s: %s", model_name, exc)
                return None

        predictor = self._cache[cache_key]
        predictor.conf_threshold = conf_threshold
        return predictor

    def load_from_ui(
        self,
        file_path: Optional[str],
        model_name: Optional[str],
        use_predefined: Optional[str] = None,
        conf_threshold: float = 0.25,
    ) -> Optional[DetectionPredictor]:
        """Load model based on UI component states.

        Args:
            file_path: Path from file upload component.
            model_name: Selected model name from dropdown.
            use_predefined: Radio button value ("Predefined" or "Upload File").
                If None, auto-detects based on which component has a value.
            conf_threshold: Detection confidence threshold.

        Returns:
            DetectionPredictor instance, or None if loading fails.
        """
        # Determine which source to use
        if use_predefined == "Predefined" or (
            use_predefined is None and model_name and not file_path
        ):
            return self.load_predefined(model_name, conf_threshold)
        elif use_predefined == "Upload File" or (
            use_predefined is None and file_path
        ):
            return self.load_from_file(file_path, conf_threshold)
        else:
            return None

    def _default_data_config(self) -> Dict:
        """Get default COCO data config from app_demo/config/coco_names.yaml."""
        return load_coco_data_config(self._config)
