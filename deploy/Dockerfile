FROM python:3.11-slim

# Install system dependencies for OpenCV and audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    portaudio19-dev \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for layer caching.
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source.
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Expose the app. No network ports — all I/O is local.
# Camera access requires --device /dev/video0.
CMD ["python", "-m", "cruze", "--hardware", "desktop"]
