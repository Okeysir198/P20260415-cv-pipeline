FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip curl git \
    libgl1-mesa-dri libglib2.0-0 libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml .
RUN /root/.local/bin/uv sync

COPY . .

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 18100

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:18100/health')" || exit 1

CMD ["/root/.local/bin/uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18100"]
