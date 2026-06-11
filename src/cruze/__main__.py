"""
Entry point: python -m cruze

Examples:
  python -m cruze                          # desktop profile, real camera
  python -m cruze --hardware desktop --no-camera   # simulated frames
  python -m cruze --hardware jetson_orin   # production Jetson config
  python -m cruze --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cruze",
        description="Cruze — vision-first AI co-pilot for the car dashboard",
    )
    p.add_argument(
        "--hardware",
        default="desktop",
        choices=["desktop", "jetson_orin", "pi5"],
        help="Hardware profile to load (default: desktop)",
    )
    p.add_argument(
        "--no-camera",
        action="store_true",
        help="Use simulated frames instead of a real camera device",
    )
    p.add_argument(
        "--video",
        default=None,
        metavar="PATH",
        help="Replay a video file instead of a live camera (paced at the file's FPS)",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="With --video, restart the file when it ends instead of stopping",
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override the logging level from config",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from cruze.orchestrator import build_and_run
    build_and_run(
        hardware_profile=args.hardware,
        no_camera=args.no_camera,
        log_level=args.log_level,
        video=args.video,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
