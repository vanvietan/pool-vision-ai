"""Ball + pocket detection.

`Detector` is the seam for swapping detection backends. `OpenCVDetector`
uses classic CV (Hough circles + color). `YoloDetector` uses an ultralytics
YOLO model — same interface, no pipeline changes. `make_detector()` picks the
backend from env and falls back to OpenCV when YOLO is unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Protocol

import cv2
import numpy as np

from .models import Ball, Pocket
from .table import OUT_H, OUT_W, _cloth_mask

log = logging.getLogger(__name__)


def table_pockets() -> List[Pocket]:
    """6 pockets at fixed positions in the normalized warped rectangle."""
    return [
        Pocket(name="top_left", x=0, y=0),
        Pocket(name="mid_left", x=OUT_W / 2, y=0),
        Pocket(name="top_right", x=OUT_W - 1, y=0),
        Pocket(name="bottom_left", x=0, y=OUT_H - 1),
        Pocket(name="mid_right", x=OUT_W / 2, y=OUT_H - 1),
        Pocket(name="bottom_right", x=OUT_W - 1, y=OUT_H - 1),
    ]


class Detector(Protocol):
    def detect(self, warped: np.ndarray) -> List[Ball]:
        ...


# --- shared cue-ball classification (used by every backend) -----------------

def _mean_hsv(hsv: np.ndarray, x: int, y: int, r: int):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), max(1, int(r * 0.7)), 255, -1)
    return cv2.mean(hsv, mask=mask)[:3]


def _mark_cue_ball(balls: List[Ball]) -> None:
    """Cue ball = the white ball; every other (colored) ball is a target.

    The cue is always present in play, so the whitest ball (high value, low
    saturation) is always marked as the cue — no hard white gate, which would
    drop the cue under poor lighting and leave no shot to plan.
    """
    if not balls:
        return

    def whiteness(b: Ball) -> float:
        _, s, v = b.color_hsv or [0.0, 255.0, 0.0]
        return v - s  # high value, low saturation

    max(balls, key=whiteness).is_cue = True


class OpenCVDetector:
    """Color-blob ball detector.

    Balls are the round objects sitting *on* the cloth, so instead of fragile
    Hough circles this masks the felt, keeps only the largest cloth region as
    the playing area, and treats the non-cloth blobs inside it as balls. This
    ignores off-table clutter (floor, shoes, rails) and reliably finds colored
    balls that Hough often misses.
    """

    def __init__(self, min_radius: int = 7, max_radius: int = 40):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, warped: np.ndarray) -> List[Ball]:
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        cloth = _cloth_mask(hsv)

        # playing area = largest filled cloth contour, eroded to drop the rails
        cnts, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        area = np.zeros(cloth.shape, np.uint8)
        cv2.drawContours(area, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        area = cv2.erode(area, np.ones((9, 9), np.uint8))

        # balls = non-cloth pixels inside the playing area
        objs = cv2.bitwise_and(area, cv2.bitwise_not(cloth))
        objs = cv2.morphologyEx(objs, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        objs = cv2.morphologyEx(objs, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        n, _, stats, centroids = cv2.connectedComponentsWithStats(objs)
        min_a = np.pi * self.min_radius ** 2
        max_a = np.pi * self.max_radius ** 2

        balls: List[Ball] = []
        for i in range(1, n):  # 0 is background
            a = stats[i, cv2.CC_STAT_AREA]
            if not (min_a <= a <= max_a):
                continue
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # round-ish: bbox near-square and well filled (rejects streaks/glare)
            if min(bw, bh) == 0 or max(bw, bh) / min(bw, bh) > 1.8:
                continue
            if a / float(bw * bh) < 0.55:
                continue
            cx, cy = centroids[i]
            r = float(np.sqrt(a / np.pi))
            h, s, v = _mean_hsv(hsv, int(cx), int(cy), int(r))
            balls.append(
                Ball(id=len(balls), x=float(cx), y=float(cy), radius=r,
                     color_hsv=[float(h), float(s), float(v)])
            )

        _mark_cue_ball(balls)
        return balls


class YoloDetector:
    """Ultralytics YOLO ball detector.

    Detects balls as bounding boxes, converts each to a circle (center +
    radius), then reuses the shared HSV cue-ball classifier. Defaults to the
    COCO `sports ball` class (32) so a stock `yolov8n.pt` works out of the box;
    point `YOLO_MODEL` at a pool-specific model for the PRD >95% target and set
    `YOLO_CLASSES` to that model's ball class ids.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
    ):
        from ultralytics import YOLO  # lazy: heavy torch import

        self.model = YOLO(model_path)
        self.conf = conf
        # COCO class 32 == "sports ball"; override for custom pool models.
        self.classes = classes if classes is not None else [32]

    def detect(self, warped: np.ndarray) -> List[Ball]:
        res = self.model.predict(
            warped, conf=self.conf, classes=self.classes, verbose=False
        )
        balls: List[Ball] = []
        if not res:
            return balls

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        boxes = res[0].boxes
        for i, xyxy in enumerate(boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = xyxy
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            r = float(min(x2 - x1, y2 - y1) / 2)
            h, s, v = _mean_hsv(hsv, int(cx), int(cy), int(r))
            balls.append(
                Ball(id=i, x=cx, y=cy, radius=r,
                     color_hsv=[float(h), float(s), float(v)])
            )

        _mark_cue_ball(balls)
        return balls


def make_detector() -> Detector:
    """Pick a detector from env; fall back to OpenCV when YOLO is unavailable.

    Env:
      DETECTOR     opencv (default) | yolo
      YOLO_MODEL   model path/name (default yolov8n.pt)
      YOLO_CONF    confidence threshold (default 0.25)
      YOLO_CLASSES comma-separated class ids (default 32 = COCO sports ball)
    """
    backend = os.getenv("DETECTOR", "opencv").lower()
    if backend != "yolo":
        return OpenCVDetector()

    try:
        classes_env = os.getenv("YOLO_CLASSES")
        classes = (
            [int(c) for c in classes_env.split(",") if c.strip()]
            if classes_env
            else None
        )
        det = YoloDetector(
            model_path=os.getenv("YOLO_MODEL", "yolov8n.pt"),
            conf=float(os.getenv("YOLO_CONF", "0.25")),
            classes=classes,
        )
        log.info("using YoloDetector")
        return det
    except Exception as e:  # missing ultralytics/torch, bad model, etc.
        log.warning("YOLO unavailable (%s); falling back to OpenCVDetector", e)
        return OpenCVDetector()
