"""
Record a synced camera + telemetry session for offline replay.

Output layout:
  drives/<timestamp>/
    frames/  0000000.jpg  0000001.jpg ...
    telemetry.jsonl       one JSON object per line, timestamped

Replay:
  Feed frames/ as a camera source and replay telemetry.jsonl into the
  VehicleState channel to develop reasoning/personality offline.

Usage:
  python scripts/record_drive.py --output drives/my_session
  python scripts/record_drive.py --device 0 --fps 10 --duration 300
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))


def main() -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python required: pip install 'cruze[vision]'")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--output", default=f"drives/{int(time.time())}")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--duration", type=float, default=None, help="Seconds to record (None = until Q)")
    args = p.parse_args()

    out = pathlib.Path(args.output)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out / "telemetry.jsonl"

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    interval = 1.0 / args.fps
    frame_id = 0
    t_start = time.monotonic()

    print(f"Recording to {out}  (Q to stop)")
    with telemetry_path.open("w") as tel_f:
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                break

            fname = frames_dir / f"{frame_id:07d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            record = {"frame_id": frame_id, "timestamp": time.time()}
            tel_f.write(json.dumps(record) + "\n")
            frame_id += 1

            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if args.duration and (time.monotonic() - t_start) >= args.duration:
                break

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    cap.release()
    cv2.destroyAllWindows()
    print(f"Recorded {frame_id} frames to {out}")


if __name__ == "__main__":
    main()
