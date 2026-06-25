"""Shot analysis engine (ghost-ball method).

For each (object ball, pocket) pair it computes the ghost-ball aim point, the
cut angle, checks for blockers, and scores difficulty. Returns the best
unblocked shot plus all ranked candidates.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .models import Ball, Pocket, Shot, Spin

# Weights for the difficulty heuristic.
W_ANGLE = 0.55
W_DIST = 0.45
MAX_DIST = 1200.0  # ~diagonal of the normalized table, for distance normalization


def _v(p) -> np.ndarray:
    return np.array(p, dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Shortest distance from point p to segment ab."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _blocked(seg_a, seg_b, blockers: List[Ball], r: float, ignore_ids) -> bool:
    for b in blockers:
        if b.id in ignore_ids:
            continue
        if _point_seg_dist(_v([b.x, b.y]), seg_a, seg_b) < (r + b.radius) * 0.9:
            return True
    return False


def _recommend_spin(d_cue: float, d_obj: float) -> Spin:
    """Basic vertical-English aid (stun / follow / draw).

    Aim itself comes from the ghost-ball line, so side spin is left at center
    for accuracy; only top/bottom is suggested to control the cue ball after
    contact, based on how far the whole shot travels.
    """
    frac = (d_cue + d_obj) / MAX_DIST
    if frac >= 0.55:
        return Spin(
            hit_x=0.0, hit_y=-0.5, zone="Top/Follow",
            tip="Long shot — strike above center so the cue ball carries through.",
        )
    if frac <= 0.30:
        return Spin(
            hit_x=0.0, hit_y=0.5, zone="Draw",
            tip="Short shot — strike below center to pull the cue ball back and avoid a scratch.",
        )
    return Spin(
        hit_x=0.0, hit_y=0.0, zone="Stun/Center",
        tip="Strike dead center for an accurate, controlled stop.",
    )


def analyze_shots(
    cue: Optional[Ball],
    balls: List[Ball],
    pockets: List[Pocket],
    target_ball: Optional[int] = None,
) -> Tuple[Optional[Shot], List[Shot]]:
    if cue is None:
        return None, []

    cue_p = _v([cue.x, cue.y])
    r = cue.radius
    objects = [b for b in balls if not b.is_cue]
    candidates: List[Shot] = []

    for ob in objects:
        ob_p = _v([ob.x, ob.y])
        for pk in pockets:
            pk_p = _v([pk.x, pk.y])

            to_pocket = _unit(pk_p - ob_p)
            ghost = ob_p - to_pocket * (2 * r)  # cue center target

            cue_to_ghost = ghost - cue_p
            # reject if the object ball is behind the cut (cue would push wrong way)
            if np.dot(_unit(cue_to_ghost), to_pocket) <= 0:
                continue

            cut_cos = float(np.dot(_unit(cue_to_ghost), to_pocket))
            cut_cos = max(-1.0, min(1.0, cut_cos))
            cut_angle = math.degrees(math.acos(cut_cos))

            d_cue = float(np.linalg.norm(cue_to_ghost))
            d_obj = float(np.linalg.norm(pk_p - ob_p))

            blocked = _blocked(cue_p, ghost, objects, r, {ob.id}) or _blocked(
                ob_p, pk_p, objects, r, {ob.id}
            )
            if blocked:
                continue

            angle_pen = 1.0 - cut_cos  # 0 straight .. up to 2 at 180
            dist_pen = min(1.0, (d_cue + d_obj) / MAX_DIST)
            difficulty = max(0.0, min(1.0, W_ANGLE * angle_pen + W_DIST * dist_pen))
            success = max(0.0, min(1.0, (cut_cos ** 2) * (1.0 - dist_pen)))

            contact = ob_p - to_pocket * r  # surface contact point on object ball

            candidates.append(
                Shot(
                    target_ball=ob.id,
                    target_pocket=pk.name,
                    difficulty=round(difficulty, 3),
                    success_rate=round(success, 3),
                    cue=cue_p.tolist(),
                    ghost=ghost.tolist(),
                    contact=contact.tolist(),
                    object_center=ob_p.tolist(),
                    pocket=pk_p.tolist(),
                    spin=_recommend_spin(d_cue, d_obj),
                )
            )

    candidates.sort(key=lambda s: s.difficulty)
    # best = easiest shot overall, or the easiest for a requested target ball
    pool = (
        [c for c in candidates if c.target_ball == target_ball]
        if target_ball is not None
        else candidates
    )
    best = pool[0] if pool else None
    return best, candidates
