"""Pure-numpy preprocessing for the vision_pilot ONNX nets, plus two thin
cv2-backed spatial transforms. All coordinate/decoding math is numpy-only so
it is unit-tested without cv2 or onnxruntime (CLAUDE.md testing rule)."""

from __future__ import annotations

import numpy as np

# vision_pilot net input size (see AutoSpeed/AutoSteer/AutoDrive headers).
NET_W = 1024
NET_H = 512

# ImageNet normalisation — AutoDrive only (inference.cpp chw_imagenet).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def top_crop_2_1(h: int, w: int) -> int:
    """Rows to drop from the top so the kept region is 2:1 (w:h) — drops sky,
    keeps the bottom w/2 rows. Matches vision_pilot compute_top_crop_2_1()."""
    return max(0, round(h - w / 2.0))


def crop_resize_params(h: int, w: int) -> tuple[int, float, float]:
    """(crop_top, sx, sy) — the scale factors that map resized 1024x512 px
    back to raw px after crop-2:1 + resize."""
    crop_top = top_crop_2_1(h, w)
    sx = w / NET_W
    sy = (h - crop_top) / NET_H
    return crop_top, sx, sy


def unmap_point(x: float, y: float, sx: float, sy: float, crop_top: int) -> tuple[float, float]:
    """Resized-space (1024x512) pixel → raw-frame pixel."""
    return x * sx, y * sy + crop_top


def chw_from_rgb01(rgb01: np.ndarray, imagenet: bool) -> np.ndarray:
    """HWC float RGB in [0,1] → contiguous [1,3,H,W] float32, optional ImageNet norm."""
    arr = rgb01.astype(np.float32, copy=False)
    if imagenet:
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None])


def decode_yolo(raw: np.ndarray, conf_thres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLO decode of [1, 4+K, N] → (boxes[M,4] xyxy, scores[M], class_ids[M])
    in net px. Mirrors auto_speed.cpp post_process (sigmoid + argmax + threshold)."""
    data = raw[0]                       # [C, N]
    cx, cy, w, h = data[0], data[1], data[2], data[3]
    logits = data[4:]                   # [K, N]
    probs = 1.0 / (1.0 + np.exp(-logits))
    class_ids = np.argmax(probs, axis=0)
    scores = probs[class_ids, np.arange(probs.shape[1])]
    keep = scores >= conf_thres
    boxes = np.stack(
        [cx[keep] - w[keep] / 2, cy[keep] - h[keep] / 2,
         cx[keep] + w[keep] / 2, cy[keep] + h[keep] / 2],
        axis=1,
    ).astype(np.float32)
    return boxes, scores[keep].astype(np.float32), class_ids[keep].astype(np.int64)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS → kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)  # epsilon avoids 0/0 for zero-area boxes
        order = order[1:][iou <= iou_thres]
    return keep


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without cv2
        raise ImportError(
            "opencv-python required for image preprocessing. "
            "Install with: pip install 'cruze[vision]'"
        ) from exc
    return cv2


def preprocess_crop2_1(image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Raw BGR frame → ([1,3,512,1024] RGB[0,1] CHW, sx, sy, crop_top).
    Shared input for AutoSpeed + AutoSteer."""
    cv2 = _require_cv2()
    h, w = image_bgr.shape[:2]
    crop_top, sx, sy = crop_resize_params(h, w)
    cropped = image_bgr[crop_top:h, 0:w]
    resized = cv2.resize(cropped, (NET_W, NET_H), interpolation=cv2.INTER_LINEAR)
    rgb01 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # uint8 → [0,1]
    return chw_from_rgb01(rgb01, imagenet=False), sx, sy, crop_top


def preprocess_bev(image_bgr: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Raw BGR frame → BEV-warped [1,3,512,1024] CHW, ImageNet-normed. AutoDrive input."""
    cv2 = _require_cv2()
    warped = cv2.warpPerspective(
        image_bgr, homography, (NET_W, NET_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    rgb01 = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # uint8 → [0,1]
    return chw_from_rgb01(rgb01, imagenet=True)
