"""Table detection + perspective correction.

Finds the green felt playing surface, locates its 4 corners and warps the
image to a fixed top-down rectangle. Falls back to using the raw image when
the table cannot be found, so the pipeline still produces a result.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Fixed normalized table size (2:1, roughly a 9ft pool table ratio).
OUT_W = 1000
OUT_H = 500

# Destination rectangle corners (TL, TR, BR, BL) for every warp.
_DST = np.array(
    [[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
    dtype=np.float32,
)

# Wide green→cyan→blue range used as a fallback when auto-detection of the
# cloth color is inconclusive. Pool cloth is typically green or blue.
LOWER_CLOTH = np.array([35, 40, 40])
UPPER_CLOTH = np.array([130, 255, 255])


def _cloth_mask(hsv: np.ndarray) -> np.ndarray:
    """Mask of the playing cloth, auto-detecting its dominant hue.

    Hardcoded green fails on blue (or red) cloth. Instead find the dominant
    saturated, mid-bright hue (the felt fills most of the frame) and build a
    range around it. Falls back to a wide green→blue range when too few
    saturated pixels are found.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # cloth is saturated and mid-bright: excludes grey walls/floor, white
    # balls (low sat), dark wood rails and shadows (low val).
    cand = (s > 60) & (v > 40) & (v < 245)
    n = hsv.shape[0] * hsv.shape[1]
    if int(cand.sum()) < 0.05 * n:
        return cv2.inRange(hsv, LOWER_CLOTH, UPPER_CLOTH)

    dom = int(np.argmax(np.bincount(h[cand].ravel(), minlength=180)))
    lower = np.array([max(0, dom - 15), 50, 40])
    upper = np.array([min(179, dom + 15), 255, 255])
    return cv2.inRange(hsv, lower, upper)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _valid_quad(src: np.ndarray, img_area: float) -> bool:
    """Reject degenerate table quads before trusting the warp.

    A steep oblique photo often yields a collapsed/triangular quad whose warp
    is garbage. Require: 4 distinct corners, convex, sane area, and a roughly
    landscape aspect (a pool table is ~2:1, never portrait).
    """
    # corners distinct (no two within 3% of the image diagonal)
    diag = float(np.hypot(*_quad_span(src)))
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(src[i] - src[j]) < 0.03 * diag:
                return False
    if not cv2.isContourConvex(src.astype(np.int32)):
        return False
    area = cv2.contourArea(src.astype(np.float32))
    if area < 0.15 * img_area:
        return False
    w = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
    h = (np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2
    if h < 1e-6 or not (1.2 <= w / h <= 4.0):  # landscape, near 2:1
        return False
    return True


def _quad_span(src: np.ndarray) -> Tuple[float, float]:
    xs, ys = src[:, 0], src[:, 1]
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def warp_from_corners(
    img: np.ndarray, corners: Sequence[Sequence[float]]
) -> np.ndarray:
    """Warp using user-supplied corners (TL, TR, BR, BL), normalized to [0,1].

    Values may fall outside [0,1] when a corner is off-frame or occluded; the
    homography still resolves and `warpPerspective` samples outside the image
    as black. The most reliable path for oblique photos.
    """
    h, w = img.shape[:2]
    src = np.array(
        [[float(x) * w, float(y) * h] for x, y in corners], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(src, _DST)
    return cv2.warpPerspective(img, m, (OUT_W, OUT_H))


def detect_table(
    img: np.ndarray, corners: Optional[Sequence[Sequence[float]]] = None
) -> Tuple[np.ndarray, bool]:
    """Return (warped BGR image of size OUT_HxOUT_W, table_detected).

    If `corners` (4 normalized TL,TR,BR,BL points) are supplied they are used
    directly — the reliable path for oblique/real photos. Otherwise auto-detect
    the cloth and warp; `table_detected=False` means the warp could not be
    trusted and the raw image was resized as a fallback (pocket geometry then
    unreliable, surface to the user).
    """
    if corners is not None and len(corners) == 4:
        return warp_from_corners(img, corners), True

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _cloth_mask(hsv)
    # close gaps from balls/glare sitting on the cloth
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 0.15 * img_area:
            # convex hull → 4-point approximation is more stable than raw
            # approxPolyDP on a ragged cloth contour.
            hull = cv2.convexHull(biggest)
            peri = cv2.arcLength(hull, True)
            for eps in np.arange(0.01, 0.12, 0.01):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    src = _order_corners(approx)
                    if _valid_quad(src, img_area):
                        m = cv2.getPerspectiveTransform(src, _DST)
                        warped = cv2.warpPerspective(img, m, (OUT_W, OUT_H))
                        return warped, True
                    break

    # fallback: resize raw image into the normalized canvas (geometry unreliable)
    warped = cv2.resize(img, (OUT_W, OUT_H))
    return warped, False
