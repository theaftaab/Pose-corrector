import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import app
from services.postureAnalyzer import (
    AnalysisResult,
    analyze,
    angle_from_horizontal,
    angle_from_vertical,
    detect_view,
)

API_HEADERS = {"X-API-Key": "12345678-unsafe-master-key"}


def _blank_kp() -> np.ndarray:
    """Returns a (17, 3) array with all zeros and zero confidence."""
    return np.zeros((17, 3), dtype=np.float32)


def _set(kp: np.ndarray, idx: int, x: float, y: float, conf: float = 0.9) -> None:
    kp[idx] = (x, y, conf)


# COCO indices reused by the tests (kept local to avoid coupling to constants).
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12


def _front_pose(image_w: int = 1000, shoulder_dx: int = 200) -> np.ndarray:
    kp = _blank_kp()
    cx = image_w // 2
    _set(kp, NOSE, cx, 100)
    _set(kp, L_EAR, cx - 30, 110)
    _set(kp, R_EAR, cx + 30, 110)
    _set(kp, L_SH, cx - shoulder_dx // 2, 200)
    _set(kp, R_SH, cx + shoulder_dx // 2, 200)
    _set(kp, L_HIP, cx - 80, 400)
    _set(kp, R_HIP, cx + 80, 400)
    return kp


def _side_pose(image_w: int = 1000) -> np.ndarray:
    kp = _blank_kp()
    cx = image_w // 2
    _set(kp, NOSE, cx + 30, 100)
    _set(kp, R_EAR, cx, 110)              # only one ear visible
    _set(kp, L_SH, cx, 200)               # shoulders nearly stacked
    _set(kp, R_SH, cx + 5, 200)
    _set(kp, L_HIP, cx, 400)
    _set(kp, R_HIP, cx + 5, 400)
    return kp


class TestAngleHelpers(unittest.TestCase):
    def test_horizontal_level(self):
        self.assertAlmostEqual(angle_from_horizontal((0, 100), (200, 100)), 0.0, places=3)

    def test_horizontal_tilt_right_shoulder_up(self):
        # Right point above left in screen terms -> positive angle.
        a = angle_from_horizontal((0, 100), (100, 90))
        self.assertGreater(a, 0)
        self.assertLess(a, 10)

    def test_vertical_zero_when_aligned(self):
        self.assertAlmostEqual(angle_from_vertical((100, 50), (100, 200)), 0.0, places=3)

    def test_vertical_lean_forward(self):
        # Top point shifted in +x => positive angle from vertical.
        a = angle_from_vertical((130, 50), (100, 200))
        self.assertGreater(a, 0)
        self.assertLess(a, 30)


class TestViewDetection(unittest.TestCase):
    def test_front_view_detected(self):
        self.assertEqual(detect_view(_front_pose(), image_width=1000), "front")

    def test_side_view_detected(self):
        self.assertEqual(detect_view(_side_pose(), image_width=1000), "side")

    def test_unknown_when_no_shoulders(self):
        self.assertEqual(detect_view(_blank_kp(), image_width=1000), "unknown")


class TestAnalyzeFront(unittest.TestCase):
    def test_level_shoulders_classify_ok(self):
        kp = _front_pose()
        result = analyze(kp, image_width=1000)
        self.assertEqual(result.view, "front")
        self.assertTrue(result.person_detected)
        regions = {m.region: m for m in result.metrics}
        self.assertIn("shoulder_tilt", regions)
        self.assertEqual(regions["shoulder_tilt"].status, "ok")

    def test_uneven_shoulders_flag_bad(self):
        kp = _front_pose()
        kp[R_SH, 1] = 175  # right shoulder noticeably higher
        result = analyze(kp, image_width=1000)
        regions = {m.region: m for m in result.metrics}
        self.assertEqual(regions["shoulder_tilt"].status, "bad")

    def test_lateral_lean_flagged(self):
        kp = _front_pose()
        # Push shoulders horizontally relative to hips to create a lean.
        kp[L_SH, 0] += 60
        kp[R_SH, 0] += 60
        result = analyze(kp, image_width=1000)
        regions = {m.region: m for m in result.metrics}
        self.assertIn("lateral_lean", regions)
        self.assertIn(regions["lateral_lean"].status, {"warn", "bad"})


class TestAnalyzeSide(unittest.TestCase):
    def test_side_metrics_present(self):
        kp = _side_pose()
        result = analyze(kp, image_width=1000)
        self.assertEqual(result.view, "side")
        regions = {m.region: m for m in result.metrics}
        self.assertIn("neck_inclination", regions)
        self.assertIn("torso_inclination", regions)

    def test_forward_head_flagged(self):
        kp = _side_pose()
        kp[R_EAR, 0] += 60  # ear well forward of shoulder
        result = analyze(kp, image_width=1000)
        regions = {m.region: m for m in result.metrics}
        self.assertIn(regions["neck_inclination"].status, {"warn", "bad"})


class TestAnalyzeEmpty(unittest.TestCase):
    def test_no_keypoints_marks_no_person(self):
        result = analyze(None, image_width=1000)
        self.assertFalse(result.person_detected)
        self.assertEqual(result.view, "unknown")
        self.assertEqual(result.metrics, [])


class TestPoseEndpoint(unittest.TestCase):
    """Integration tests for /api/v1/pose with the YOLO model mocked out."""

    def setUp(self):
        self.client = TestClient(app)

    def _jpeg_bytes(self, w: int = 640, h: int = 480) -> bytes:
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        return encoded.tobytes()

    def test_rejects_non_image_upload(self):
        files = {"image": ("notes.txt", b"hello world", "text/plain")}
        response = self.client.post("/api/v1/pose", files=files, headers=API_HEADERS)
        self.assertEqual(400, response.status_code)

    @patch("services.poseService.pose_service.run")
    def test_returns_envelope_with_metrics(self, mock_run):
        mock_run.return_value = (
            AnalysisResult(view="front", person_detected=True, metrics=[]),
            self._jpeg_bytes(),
        )
        files = {"image": ("frame.jpg", self._jpeg_bytes(), "image/jpeg")}
        response = self.client.post("/api/v1/pose", files=files, headers=API_HEADERS)
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual(body["view"], "front")
        self.assertTrue(body["person_detected"])
        self.assertIsInstance(body["metrics"], list)
        self.assertTrue(body["image_base64"])


if __name__ == "__main__":
    unittest.main()
