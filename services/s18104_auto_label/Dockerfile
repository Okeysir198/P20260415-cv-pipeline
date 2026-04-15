# Use Python slim base (has Python 3.12, pip, and basic tools)
FROM python:3.12-slim

# Install minimal system dependencies for opencv-python-headless
# Avoid libgl1-mesa-dri which pulls in entire graphics stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Minimal GL libraries (avoid full Mesa stack)
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Install dependencies (uv handles Python caching efficiently)
COPY s18104_auto_label/pyproject.toml .
RUN /root/.local/bin/uv sync

# Copy shared code and source
COPY _shared/ /app/_shared/
COPY s18104_auto_label/ .

EXPOSE 18104

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:18104/health')" || exit 1

CMD ["/root/.local/bin/uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18104"]
