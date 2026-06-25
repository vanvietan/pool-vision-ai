"""Pydantic data models shared across the AI service."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Ball(BaseModel):
    id: int
    x: float
    y: float
    radius: float
    is_cue: bool = False
    # mean HSV inside the ball, useful for debugging classification
    color_hsv: Optional[List[float]] = None


class Pocket(BaseModel):
    name: str  # top_left, top_right, mid_left, mid_right, bottom_left, bottom_right
    x: float
    y: float


class Spin(BaseModel):
    """Where to strike the cue ball (a basic stun/follow/draw aid).

    hit_x/hit_y are offsets from the cue-ball center as a fraction of its
    radius: x in [-1 (left), +1 (right)], y in [-1 (top), +1 (bottom)].
    """
    hit_x: float
    hit_y: float
    zone: str  # matches the 9-zone chart label, e.g. "Top/Follow", "Draw"
    tip: str   # one-line plain-language coaching note


class Power(BaseModel):
    """How hard to hit, derived from shot distance and cut angle."""
    level: float  # 0 (feather) .. 1 (full break-speed)
    label: str    # Soft / Medium / Firm / Break
    tip: str      # one-line plain-language note


class ObjectHit(BaseModel):
    """Where the cue ball strikes the object ball (ghost-ball contact).

    hit_x/hit_y are offsets from the object-ball center as a fraction of its
    radius (x right, y down) pointing to the surface contact spot.
    """
    hit_x: float
    hit_y: float
    fullness: str  # Full / Three-quarter / Half / Thin
    cut_angle: float  # degrees, 0 = dead straight
    tip: str


class Shot(BaseModel):
    target_ball: int
    target_pocket: str
    difficulty: float  # 0 (easy) .. 1 (hard)
    success_rate: float  # 0 .. 1
    # geometry used to draw the overlay (normalized/warped coordinates)
    cue: List[float]
    ghost: List[float]  # aim point: where cue ball center must arrive
    contact: List[float]  # contact point on the object ball surface
    object_center: List[float]
    pocket: List[float]
    spin: Optional[Spin] = None  # recommended cue-ball strike point
    power: Optional[Power] = None  # how hard to hit
    object_hit: Optional[ObjectHit] = None  # where to strike the object ball


class AnalyzeResult(BaseModel):
    table_detected: bool
    width: int
    height: int
    cue_ball: Optional[Ball]
    balls: List[Ball]
    pockets: List[Pocket]
    shot: Optional[Shot]
    # clean warped table (base64 PNG) so the client can redraw the overlay
    # itself when the user switches the target ball
    warped_png: str
    # all candidate shots ranked, for debugging / training mode
    candidates: List[Shot]
    overlay_png: str  # base64-encoded PNG of the warped table with overlay
