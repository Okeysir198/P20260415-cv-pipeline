"""Main Gradio application factory for the comprehensive safety detection demo.

Creates a multi-tab Gradio Blocks application. Tabs are loaded dynamically
from the ``tabs`` list in config.yaml — no hardcoded imports.

Usage:
    from app_demo.app import create_app
    app = create_app(config)
    app.launch()
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Dict

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root

from app_demo.model_manager import ModelManager

logger = logging.getLogger(__name__)


def _load_builder(dotted_path: str):
    """Import a tab builder function from a dotted module path.

    E.g. ``app_demo.tabs.tab_fire.build_tab_fire`` -> the function object.
    """
    module_path, func_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def create_app(config: Dict) -> gr.Blocks:
    """Build the Gradio Blocks application with tabs from config.

    Args:
        config: Full demo config dict loaded from ``app_demo/config.yaml``.

    Returns:
        ``gr.Blocks`` application instance (call ``.launch()`` to start).
    """
    gradio_config = config.get("gradio", {})
    title = gradio_config.get("title", "Safety Detection Demo")

    manager = ModelManager(config)
    manager.warmup()

    tab_configs = config.get("tabs", [])

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "Comprehensive safety detection for industrial environments. "
            "Select a tab to get started with a specific use case."
        )

        with gr.Tabs():
            for tab_cfg in tab_configs:
                builder = _load_builder(tab_cfg["builder"])
                builder(manager, config)

    logger.info("Built Gradio app with %d tabs", len(tab_configs))
    return app
