"""
Per-module latency benchmark.

Runs each pipeline stage on synthetic data and reports mean ± stddev latency
for detector, tracker, depth, and full perception pipeline.

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --backend stub --frames 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from cruze.core.types import BBox, Detection, Frame, ObjectClass
from cruze.perception import depth as depth_mod
from cruze.perception.detector import load as load_detector
from cruze.perception.tracker import Tracker


def _synthetic_frame(width: int = 1280, height: int = 720, frame_id: int = 0) -> Frame:
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    focal = depth_mod.focal_length_px(width, 70.0)
    return Frame(image=image, frame_id=frame_id, focal_length_px=focal)


def _synthetic_detections(n: int = 4) -> list[Detection]:
    return [
        Detection(
            bbox=BBox(x * 200.0, 200.0, x * 200.0 + 100, 350.0),
            confidence=0.9,
            cls=ObjectClass.CAR,
            distance_m=10.0 + x * 5,
        )
        for x in range(n)
    ]


def bench(fn, n: int) -> tuple[float, float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="stub", help="Detector backend")
    p.add_argument("--frames", type=int, default=100, help="Frames to benchmark")
    args = p.parse_args()

    print(f"\nCruze Latency Benchmark  ({args.frames} frames, backend={args.backend})\n")
    print(f"{'Module':<30} {'Mean (ms)':>10} {'Stddev (ms)':>12}")
    print("-" * 55)

    frame = _synthetic_frame()
    dets = _synthetic_detections()

    # Detector.
    detector = load_detector(args.backend, model_path="models/yolov8n.pt")
    mean, std = bench(lambda: detector.detect(frame), args.frames)
    print(f"{'Detector (' + args.backend + ')':<30} {mean:>10.2f} {std:>12.2f}")

    # Depth estimation.
    def _depth_bench():
        for d in dets:
            depth_mod.estimate_distance(d.bbox, d.cls, frame.focal_length_px)

    mean, std = bench(_depth_bench, args.frames)
    print(f"{'Depth (monocular, 4 objects)':<30} {mean:>10.3f} {std:>12.3f}")

    # Tracker.
    tracker = Tracker()
    mean, std = bench(lambda: tracker.update(dets), args.frames)
    print(f"{'Tracker (SORT, 4 tracks)':<30} {mean:>10.3f} {std:>12.3f}")

    print()


if __name__ == "__main__":
    main()
