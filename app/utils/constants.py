"""Shared constants for the pose-corrector service."""

# COCO-17 keypoint indices used by Ultralytics YOLO pose models.
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Skeleton edges drawn by the renderer (pairs of COCO-17 indices).
SKELETON_EDGES = (
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_LEFT_KNEE),
    (KP_LEFT_KNEE, KP_LEFT_ANKLE),
    (KP_RIGHT_HIP, KP_RIGHT_KNEE),
    (KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
)

KP_CONFIDENCE_THRESHOLD = 0.5

# (ok_max, warn_max). Above warn_max is "bad". Units are degrees unless noted.
POSTURE_THRESHOLDS = {
    "shoulder_tilt": (3.0, 7.0),
    "head_tilt": (5.0, 10.0),
    "lateral_lean": (3.0, 7.0),  # percent of torso height
    "neck_inclination": (15.0, 25.0),
    "torso_inclination": (5.0, 15.0),
}

# Status colors in BGR (OpenCV).
STATUS_COLORS = {
    "ok": (80, 200, 80),
    "warn": (0, 180, 230),
    "bad": (60, 60, 230),
}
