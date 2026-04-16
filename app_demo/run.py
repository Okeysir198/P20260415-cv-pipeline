"""CLI entry point for the comprehensive safety detection Gradio demo.

Usage (from project root):
    uv run app_demo/run.py
    uv run app_demo/run.py --share

Usage (from app_demo/ with its own venv):
    cd app_demo && uv sync --all-extras
    uv run run.py
    uv run run.py --config config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path (works from both root and app_demo/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Pick the idle GPU before ModelManager.warmup() loads everything onto CUDA.
from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

import gradio as gr  # noqa: E402

from utils.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Launch comprehensive safety detection Gradio demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Auto-detect config: try app_demo/config/config.yaml (from root) then config/config.yaml (from app_demo/)
    default_config = "app_demo/config/config.yaml"
    if not Path(default_config).exists() and Path("config/config.yaml").exists():
        default_config = "config/config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to demo config YAML.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server hostname (overrides config).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Server port (overrides config).",
    )
    return parser.parse_args()


def main() -> None:
    """Load config, build the Gradio app, and launch."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    config = load_config(str(config_path))
    logger.info("Loaded config from %s", config_path)

    # Extract Gradio launch settings
    gradio_config = config.get("gradio", {})
    server_name = args.server_name or gradio_config.get("server_name", "0.0.0.0")
    server_port = args.server_port or gradio_config.get("server_port", 7861)
    share = args.share or gradio_config.get("share", False)

    # Build and launch the app
    from app_demo.app import create_app

    app = create_app(config)

    logger.info(
        "Launching Gradio app on %s:%d (share=%s)",
        server_name,
        server_port,
        share,
    )

    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Citrus(),
        css="""
            .gradio-container { max-width: 1400px !important; }
            .tab-nav button { font-size: 15px !important; font-weight: 600 !important; }
        """,
    )


if __name__ == "__main__":
    main()
