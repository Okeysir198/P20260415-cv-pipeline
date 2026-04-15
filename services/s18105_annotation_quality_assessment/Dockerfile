FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY s18105_annotation_quality_assessment/pyproject.toml .
RUN /root/.local/bin/uv sync

COPY _shared/ /app/_shared/
COPY s18105_annotation_quality_assessment/ .

EXPOSE 18105

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:18105/health')" || exit 1

CMD ["/root/.local/bin/uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18105"]
