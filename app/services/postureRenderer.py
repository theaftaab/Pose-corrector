"""OpenCV drawing helpers for posture analysis output.

Consumes a list of (keypoints, AnalysisResult) pairs and the source image;
produces an annotated BGR image with per-person skeleton, measured-angle lines,
reference axes, arcs, labels, and a top-left status HUD.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from services.postureAnalyzer import AnalysisResult, MetricDraw, MetricResult
from utils.constants import (
    KP_CONFIDENCE_THRESHOLD,
    SKELETON_EDGES,
    STATUS_COLORS,
)

# One BGR color per person (cycles if > 5 people).
_PERSON_COLORS: list[tuple[int, int, int]] = [
    (255, 200,  50),   # cyan-yellow
    ( 50, 180, 255),   # orange
    ( 80, 220,  80),   # green
    (220,  80, 220),   # purple
    ( 80,  80, 255),   # red
]

_REFERENCE_COLOR = (200, 200, 200)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_ARC_RADIUS = 36
_REFERENCE_LEN = 90


def _person_color(person_id: int) -> tuple[int, int, int]:
    return _PERSON_COLORS[person_id % len(_PERSON_COLORS)]


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    n = max(1, int(length // dash))
    for i in range(n):
        if i % 2:
            continue
        a = i / n
        b = min(1.0, (i + 1) / n)
        xa = int(x1 + (x2 - x1) * a)
        ya = int(y1 + (y2 - y1) * a)
        xb = int(x1 + (x2 - x1) * b)
        yb = int(y1 + (y2 - y1) * b)
        cv2.line(img, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def _draw_skeleton(
    img: np.ndarray, kp: np.ndarray, color: tuple[int, int, int]
) -> None:
    for a, b in SKELETON_EDGES:
        if kp[a, 2] < KP_CONFIDENCE_THRESHOLD or kp[b, 2] < KP_CONFIDENCE_THRESHOLD:
            continue
        pa = (int(kp[a, 0]), int(kp[a, 1]))
        pb = (int(kp[b, 0]), int(kp[b, 1]))
        cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for i in range(kp.shape[0]):
        if kp[i, 2] < KP_CONFIDENCE_THRESHOLD:
            continue
        cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 3, color, -1, cv2.LINE_AA)


def _reference_endpoint(
    anchor: tuple[float, float], axis: str, length: int = _REFERENCE_LEN
) -> tuple[int, int]:
    ax, ay = anchor
    if axis == "vertical":
        return int(ax), int(ay - length)
    return int(ax + length), int(ay)


def _bearing_deg(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))


def _draw_arc(
    img: np.ndarray,
    center: tuple[float, float],
    ref_end: tuple[int, int],
    measured_end: tuple[float, float],
    color: tuple[int, int, int],
) -> tuple[int, int]:
    cx, cy = int(center[0]), int(center[1])
    a_ref = _bearing_deg(center, ref_end)
    a_meas = _bearing_deg(center, measured_end)

    diff = (a_meas - a_ref + 180) % 360 - 180
    start, end = (a_ref, a_ref + diff) if diff >= 0 else (a_ref + diff, a_ref)

    cv2.ellipse(
        img, (cx, cy), (_ARC_RADIUS, _ARC_RADIUS), 0, start, end, color, 2, cv2.LINE_AA
    )

    mid_deg = (start + end) / 2
    lx = int(cx + (_ARC_RADIUS + 12) * math.cos(math.radians(mid_deg)))
    ly = int(cy + (_ARC_RADIUS + 12) * math.sin(math.radians(mid_deg)))
    return lx, ly


def _draw_metric(img: np.ndarray, metric: MetricResult) -> None:
    if metric.draw is None:
        return
    color = STATUS_COLORS[metric.status]
    d: MetricDraw = metric.draw

    a = (int(d.anchor_a[0]), int(d.anchor_a[1]))
    b = (int(d.anchor_b[0]), int(d.anchor_b[1]))
    ref = _reference_endpoint(d.anchor_a, d.reference_axis)

    _draw_dashed_line(img, a, ref, _REFERENCE_COLOR, thickness=2)
    cv2.line(img, a, b, color, 3, cv2.LINE_AA)

    label_anchor = _draw_arc(img, d.anchor_a, ref, d.anchor_b, color)
    label = f"{abs(metric.value):.1f}{_ascii(metric.unit)} (target {_ascii(metric.target)})"
    _draw_text_with_bg(img, label, label_anchor, color)


def _ascii(text: str) -> str:
    return text.replace("°", "deg").replace("≤", "<=")


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = origin
    pad = 3
    cv2.rectangle(
        img,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (30, 30, 30),
        -1,
    )
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _person_bbox(kp: np.ndarray) -> tuple[int, int, int, int] | None:
    visible = kp[kp[:, 2] >= KP_CONFIDENCE_THRESHOLD]
    if len(visible) == 0:
        return None
    xs, ys = visible[:, 0], visible[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_hud(
    img: np.ndarray,
    persons: list[tuple[np.ndarray | None, AnalysisResult]],
) -> None:
    pad = 12
    line_h = 22
    width = 300

    rows: list[tuple[str, tuple[int, int, int] | None]] = []
    for person_id, (kp, result) in enumerate(persons):
        color = _person_color(person_id)
        rows.append((f"Person {person_id + 1}  view={result.view}", color))
        for m in result.metrics:
            status_color = STATUS_COLORS[m.status]
            rows.append((
                f"  {m.region}: {abs(m.value):.1f}{_ascii(m.unit)}  [{m.status}]",
                status_color,
            ))

    height = pad * 2 + line_h * len(rows)
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + width, pad + height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    for i, (text, color) in enumerate(rows):
        y = pad + line_h * (i + 1)
        cv2.putText(img, text, (pad + 10, y), _FONT, 0.5, color or (240, 240, 240), 1, cv2.LINE_AA)


def render(
    image: np.ndarray,
    persons: list[tuple[np.ndarray | None, AnalysisResult]],
) -> np.ndarray:
    out = image.copy()
    for person_id, (kp, result) in enumerate(persons):
        color = _person_color(person_id)
        if kp is not None and kp.size > 0:
            _draw_skeleton(out, kp, color)
            bbox = _person_bbox(kp)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                label = f"#{person_id + 1}"
                cv2.putText(
                    out, label,
                    (x_min, max(14, y_min - 6)),
                    _FONT, 0.6, color, 2, cv2.LINE_AA,
                )
        for metric in result.metrics:
            _draw_metric(out, metric)
    _draw_hud(out, persons)
    return out
