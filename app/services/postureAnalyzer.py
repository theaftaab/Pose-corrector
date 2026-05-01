"""Pure posture-analysis functions over a (17, 3) keypoint array.

Layout: kp[i] = (x, y, conf), where i is a COCO-17 index. No I/O, no drawing —
the renderer consumes the structured AnalysisResult returned from analyze().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from utils.constants import (
    KP_CONFIDENCE_THRESHOLD,
    KP_LEFT_EAR,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    POSTURE_THRESHOLDS,
)

View = Literal["front", "side", "ambiguous", "unknown"]
Status = Literal["ok", "warn", "bad"]
ReferenceAxis = Literal["vertical", "horizontal"]


@dataclass
class MetricDraw:
    anchor_a: tuple[float, float]  # measured-line start
    anchor_b: tuple[float, float]  # measured-line end (also the angle vertex side)
    reference_axis: ReferenceAxis


@dataclass
class MetricResult:
    region: str
    value: float
    unit: str
    target: str
    status: Status
    draw: MetricDraw | None = None


@dataclass
class AnalysisResult:
    view: View
    person_detected: bool
    metrics: list[MetricResult] = field(default_factory=list)


# ------------------------------- helpers -------------------------------------


def _visible(kp: np.ndarray, idx: int) -> bool:
    return bool(kp[idx, 2] >= KP_CONFIDENCE_THRESHOLD)


def _xy(kp: np.ndarray, idx: int) -> tuple[float, float]:
    return float(kp[idx, 0]), float(kp[idx, 1])


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _classify(value: float, region: str) -> Status:
    ok_max, warn_max = POSTURE_THRESHOLDS[region]
    v = abs(value)
    if v <= ok_max:
        return "ok"
    if v <= warn_max:
        return "warn"
    return "bad"


def _target_label(region: str, unit: str) -> str:
    ok_max, _ = POSTURE_THRESHOLDS[region]
    return f"≤{ok_max:g}{unit}"


def angle_from_vertical(p_top: tuple[float, float], p_bottom: tuple[float, float]) -> float:
    """Signed angle (deg) between the segment p_bottom→p_top and the upward vertical.

    Positive = top point is to the right of bottom in image coords.
    Image y grows downward, so we flip dy.
    """
    dx = p_top[0] - p_bottom[0]
    dy = -(p_top[1] - p_bottom[1])
    return math.degrees(math.atan2(dx, dy))


def angle_from_horizontal(p_left: tuple[float, float], p_right: tuple[float, float]) -> float:
    """Signed angle (deg) between the segment p_left→p_right and the +x axis.

    Positive = right point is above the left point in screen terms.
    """
    dx = p_right[0] - p_left[0]
    dy = -(p_right[1] - p_left[1])
    return math.degrees(math.atan2(dy, dx))


# ------------------------------- view detection ------------------------------


def detect_view(kp: np.ndarray, image_width: int) -> View:
    if not (_visible(kp, KP_LEFT_SHOULDER) and _visible(kp, KP_RIGHT_SHOULDER)):
        return "unknown"

    l_sh = _xy(kp, KP_LEFT_SHOULDER)
    r_sh = _xy(kp, KP_RIGHT_SHOULDER)
    shoulder_spread = abs(l_sh[0] - r_sh[0]) / max(image_width, 1)

    l_ear_visible = _visible(kp, KP_LEFT_EAR)
    r_ear_visible = _visible(kp, KP_RIGHT_EAR)
    ears_visible = int(l_ear_visible) + int(r_ear_visible)

    if ears_visible == 2 and shoulder_spread > 0.15:
        return "front"
    if ears_visible <= 1 and shoulder_spread < 0.08:
        return "side"
    return "ambiguous"


# ------------------------------- metrics -------------------------------------


def shoulder_tilt(kp: np.ndarray) -> MetricResult | None:
    if not (_visible(kp, KP_LEFT_SHOULDER) and _visible(kp, KP_RIGHT_SHOULDER)):
        return None
    l = _xy(kp, KP_LEFT_SHOULDER)
    r = _xy(kp, KP_RIGHT_SHOULDER)
    angle = angle_from_horizontal(l, r)
    return MetricResult(
        region="shoulder_tilt",
        value=round(angle, 1),
        unit="°",
        target=_target_label("shoulder_tilt", "°"),
        status=_classify(angle, "shoulder_tilt"),
        draw=MetricDraw(anchor_a=l, anchor_b=r, reference_axis="horizontal"),
    )


def head_tilt(kp: np.ndarray) -> MetricResult | None:
    if not (
        _visible(kp, KP_NOSE)
        and _visible(kp, KP_LEFT_SHOULDER)
        and _visible(kp, KP_RIGHT_SHOULDER)
    ):
        return None
    nose = _xy(kp, KP_NOSE)
    sh_mid = _midpoint(_xy(kp, KP_LEFT_SHOULDER), _xy(kp, KP_RIGHT_SHOULDER))
    angle = angle_from_vertical(p_top=nose, p_bottom=sh_mid)
    return MetricResult(
        region="head_tilt",
        value=round(angle, 1),
        unit="°",
        target=_target_label("head_tilt", "°"),
        status=_classify(angle, "head_tilt"),
        draw=MetricDraw(anchor_a=sh_mid, anchor_b=nose, reference_axis="vertical"),
    )


def lateral_lean(kp: np.ndarray) -> MetricResult | None:
    needed = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP)
    if not all(_visible(kp, i) for i in needed):
        return None
    sh_mid = _midpoint(_xy(kp, KP_LEFT_SHOULDER), _xy(kp, KP_RIGHT_SHOULDER))
    hip_mid = _midpoint(_xy(kp, KP_LEFT_HIP), _xy(kp, KP_RIGHT_HIP))
    torso_height = abs(sh_mid[1] - hip_mid[1])
    if torso_height < 1.0:
        return None
    offset_pct = abs(sh_mid[0] - hip_mid[0]) / torso_height * 100.0
    return MetricResult(
        region="lateral_lean",
        value=round(offset_pct, 1),
        unit="%",
        target=_target_label("lateral_lean", "%"),
        status=_classify(offset_pct, "lateral_lean"),
        draw=MetricDraw(anchor_a=hip_mid, anchor_b=sh_mid, reference_axis="vertical"),
    )


def neck_inclination(kp: np.ndarray) -> MetricResult | None:
    """Side-view forward-head-posture: angle of shoulder→ear from vertical.

    Uses whichever ear+shoulder pair is visible (the side facing the camera).
    """
    pairs = [
        (KP_LEFT_EAR, KP_LEFT_SHOULDER),
        (KP_RIGHT_EAR, KP_RIGHT_SHOULDER),
    ]
    for ear_idx, sh_idx in pairs:
        if _visible(kp, ear_idx) and _visible(kp, sh_idx):
            ear = _xy(kp, ear_idx)
            sh = _xy(kp, sh_idx)
            angle = angle_from_vertical(p_top=ear, p_bottom=sh)
            return MetricResult(
                region="neck_inclination",
                value=round(angle, 1),
                unit="°",
                target=_target_label("neck_inclination", "°"),
                status=_classify(angle, "neck_inclination"),
                draw=MetricDraw(anchor_a=sh, anchor_b=ear, reference_axis="vertical"),
            )
    return None


def torso_inclination(kp: np.ndarray) -> MetricResult | None:
    """Side-view slouch: angle of hip→shoulder from vertical (using visible side)."""
    pairs = [
        (KP_LEFT_SHOULDER, KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    ]
    for sh_idx, hip_idx in pairs:
        if _visible(kp, sh_idx) and _visible(kp, hip_idx):
            sh = _xy(kp, sh_idx)
            hip = _xy(kp, hip_idx)
            angle = angle_from_vertical(p_top=sh, p_bottom=hip)
            return MetricResult(
                region="torso_inclination",
                value=round(angle, 1),
                unit="°",
                target=_target_label("torso_inclination", "°"),
                status=_classify(angle, "torso_inclination"),
                draw=MetricDraw(anchor_a=hip, anchor_b=sh, reference_axis="vertical"),
            )
    return None


# ------------------------------- entry point ---------------------------------


_FRONT_METRICS = (shoulder_tilt, head_tilt, lateral_lean)
_SIDE_METRICS = (neck_inclination, torso_inclination)


def analyze(kp: np.ndarray | None, image_width: int) -> AnalysisResult:
    if kp is None or kp.size == 0:
        return AnalysisResult(view="unknown", person_detected=False, metrics=[])

    view = detect_view(kp, image_width)

    if view == "front":
        candidates = _FRONT_METRICS
    elif view == "side":
        candidates = _SIDE_METRICS
    else:
        # ambiguous/unknown: try everything; only the metrics whose required
        # keypoints are visible will return a result.
        candidates = _FRONT_METRICS + _SIDE_METRICS

    metrics = [m for m in (fn(kp) for fn in candidates) if m is not None]
    return AnalysisResult(view=view, person_detected=True, metrics=metrics)
