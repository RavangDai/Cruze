# Model weights

Place model files here. They are excluded from git (see `.gitignore`).

## YOLOv8 (default desktop backend)

```bash
# Downloads automatically on first run via ultralytics:
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
# Then move the downloaded file here:
mv yolov8n-seg.pt models/
```

Or manually from https://github.com/ultralytics/assets/releases

| File | Size | Use |
|---|---|---|
| `yolov8n-seg.pt` | 7 MB | Desktop default — boxes + instance masks |
| `yolov8n.pt` | 6 MB | Boxes only (fastest; set `perception.model_path`) |
| `yolov8s.pt` | 22 MB | Boxes only, better accuracy |

Box-only weights remain fully supported: detections simply carry no
`mask_xy` and both GUIs fall back to bracket/box rendering.

## TFLite (Raspberry Pi 5 + Hailo-8)

```bash
yolo export model=yolov8n.pt format=tflite int8=True imgsz=640
mv yolov8n_saved_model/yolov8n_integer_quant.tflite models/yolov8n_hailo.tflite
```

## TensorRT (NVIDIA Jetson Orin Nano)

Run on the Jetson itself — TensorRT engines are hardware-specific:

```bash
yolo export model=yolov8n.pt format=engine device=0 half=True imgsz=640
mv yolov8n.engine models/
```

## Piper TTS

Download from https://github.com/rhasspy/piper/releases

```bash
curl -L -o models/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o models/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Whisper (faster-whisper)

Downloaded automatically by faster-whisper on first run to `~/.cache/huggingface/`.
Recommended models: `base.en` (desktop), `tiny.en` (Pi 5).

## vision_pilot ONNX weights (SP2)

AutoSpeed / AutoSteer / AutoDrive weights are vendored (Apache-2.0) in the
reference submodule. Copy them into `models/` (gitignored):

    cp reference/vision_pilot/VisionPilot/modules/models/weights/*.onnx models/

Upstream: autowarefoundation/{auto_speed,auto_steer,auto_drive}. `*_fp32.onnx`
for desktop; `*_int8.onnx` for edge (Jetson/Pi). AutoDrive additionally needs a
camera-matched homography in `calibration/vision_pilot_C.yaml`.
