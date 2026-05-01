from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from services.postureAnalyzer import AnalysisResult, analyze
from services.postureRenderer import render

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


class PoseService:
    def __init__(self, model_name: str = "yolo26m-pose.pt"):
        self.model = YOLO(ASSETS_DIR / model_name)

    def run(self, image_bytes: bytes) -> tuple[list[AnalysisResult], bytes]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")

        results = self.model(frame, verbose=False)
        all_kp = self._all_person_keypoints(results[0])

        h, w = frame.shape[:2]
        if all_kp:
            analyses = [analyze(kp, image_width=w) for kp in all_kp]
            persons = list(zip(all_kp, analyses))
        else:
            analyses = [analyze(None, image_width=w)]
            persons = [(None, analyses[0])]

        annotated = render(frame, persons)

        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Failed to encode annotated image")

        return analyses, encoded.tobytes()

    @staticmethod
    def _all_person_keypoints(result) -> list[np.ndarray]:
        """Return a list of (17, 3) arrays (x, y, conf), one per detected person."""
        kpts = getattr(result, "keypoints", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return []

        xy = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.asarray(kpts.xy)
        conf = kpts.conf.cpu().numpy() if (kpts.conf is not None and hasattr(kpts.conf, "cpu")) \
            else (np.asarray(kpts.conf) if kpts.conf is not None else np.ones(xy.shape[:2]))

        persons = []
        for i in range(xy.shape[0]):
            person_xy = xy[i]              # (17, 2)
            person_conf = conf[i][:, None] # (17, 1)
            persons.append(np.concatenate([person_xy, person_conf], axis=1))
        return persons


pose_service = PoseService()
