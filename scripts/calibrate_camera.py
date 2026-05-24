"""
Checkerboard camera calibration.

Captures frames from the configured camera, finds checkerboard corners,
and computes the camera matrix + distortion coefficients.  Saves to
calibration/camera_params.npz.

Usage:
  python scripts/calibrate_camera.py --rows 9 --cols 6 --size 25

  --rows / --cols: inner corner count on your checkerboard
  --size: physical square size in mm

Press SPACE to capture a frame, Q to quit and compute.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))


def main() -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy required: pip install 'cruze[vision]'")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=9)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--size", type=float, default=25.0, help="Square size in mm")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output", default="calibration/camera_params.npz")
    args = p.parse_args()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_pts = np.zeros((args.rows * args.cols, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.size

    world_pts = []
    img_pts = []

    cap = cv2.VideoCapture(args.device)
    print(f"Calibrating: {args.cols}x{args.rows} board, {args.size} mm squares")
    print("SPACE = capture frame | Q = finish and save")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, found)
        cv2.putText(display, f"Captures: {len(world_pts)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            world_pts.append(obj_pts)
            img_pts.append(refined)
            print(f"Captured frame {len(world_pts)}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(world_pts) < 5:
        print("Need at least 5 captures. Exiting.")
        return

    h, w = gray.shape
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_pts, img_pts, (w, h), None, None)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), camera_matrix=mtx, dist_coeffs=dist)
    print(f"Saved calibration to {out}")
    print(f"RMS reprojection error: {ret:.4f} px")
    print(f"Focal length: fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f} px")


if __name__ == "__main__":
    main()
