"""SAM3.1 Segmentation Service — entry point for uvicorn."""

from src.routes import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    from src.config import config

    server_cfg = config.get("server", {})
    uvicorn.run(
        "app:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 18106),
    )
