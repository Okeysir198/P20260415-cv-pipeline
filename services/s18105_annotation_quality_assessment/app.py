"""FastAPI annotation QA service — validates and scores ground-truth labels against SAM3 verification."""

from src.routes import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    from src.config import config

    uvicorn.run(
        "app:app",
        host=config.get("server", {}).get("host", "0.0.0.0"),
        port=config.get("server", {}).get("port", 18105),
    )
