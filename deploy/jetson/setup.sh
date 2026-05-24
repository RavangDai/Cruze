#!/usr/bin/env bash
# Jetson Orin Nano setup script.
# Run once after flashing JetPack 6.x.
# Assumes Python 3.11 and CUDA/TensorRT are installed via JetPack.

set -euo pipefail

echo "=== Cruze Jetson Setup ==="

# System packages.
sudo apt-get update
sudo apt-get install -y \
    python3.11-dev \
    libportaudio2 \
    portaudio19-dev \
    espeak-ng \
    i2c-tools  # for IMU

# Python venv.
python3.11 -m venv /opt/cruze/.venv
source /opt/cruze/.venv/bin/activate

# Install Cruze with all deps.
pip install --upgrade pip
pip install -e /opt/cruze[all]

# TensorRT Python bindings (ships with JetPack, not on PyPI).
pip install /usr/lib/python3/dist-packages/tensorrt-*.whl 2>/dev/null || true

# Convert YOLO model to TensorRT engine (run on the Jetson itself).
echo "Converting YOLOv8 to TensorRT engine..."
yolo export model=/opt/cruze/models/yolov8n.pt format=engine device=0 half=True imgsz=640
mv yolov8n.engine /opt/cruze/models/

# Install systemd service.
sudo cp /opt/cruze/deploy/systemd/cruze.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cruze
sudo systemctl start cruze

echo "=== Done. Cruze is running as a systemd service. ==="
echo "    Logs: journalctl -u cruze -f"
