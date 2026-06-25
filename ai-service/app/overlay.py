"""Render the analysis overlay on the warped table image."""
from __future__ import annotations

import base64
from typing import List, Optional

import cv2
import numpy as np

from .models import Ball, Pocket, Shot


def _pt(p) -> tuple:
    return int(round(p[0])), int(round(p[1]))


def encode_png(img: np.ndarray) -> str:
    """Base64-encode a BGR image as PNG (empty string on failure)."""
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def draw_overlay(
    warped: np.ndarray,
    balls: List[Ball],
    pockets: List[Pocket],
    shot: Optional[Shot],
) -> str:
    img = warped.copy()

    # pockets
    for pk in pockets:
        cv2.circle(img, _pt([pk.x, pk.y]), 14, (0, 0, 0), -1)
        cv2.circle(img, _pt([pk.x, pk.y]), 14, (60, 60, 60), 2)

    # balls
    for b in balls:
        color = (255, 255, 255) if b.is_cue else (0, 165, 255)
        cv2.circle(img, _pt([b.x, b.y]), int(b.radius), color, 2)
        cv2.putText(img, str(b.id), _pt([b.x - 6, b.y + 4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if shot is not None:
        cue = _pt(shot.cue)
        ghost = _pt(shot.ghost)
        obj = _pt(shot.object_center)
        pocket = _pt(shot.pocket)
        contact = _pt(shot.contact)

        # cue aim line -> ghost ball
        cv2.line(img, cue, ghost, (0, 255, 0), 2, cv2.LINE_AA)
        # object ball predicted path -> pocket
        cv2.line(img, obj, pocket, (0, 200, 255), 2, cv2.LINE_AA)
        # ghost ball position + contact point
        cv2.circle(img, ghost, 4, (0, 255, 0), -1)
        cv2.circle(img, contact, 4, (0, 0, 255), -1)
        # highlight target pocket
        cv2.circle(img, pocket, 18, (0, 255, 0), 3)

        cv2.putText(
            img,
            f"ball {shot.target_ball} -> {shot.target_pocket}  "
            f"succ {shot.success_rate:.0%}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )

    return encode_png(img)
