import io
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


class PoseService:
    def __init__(self, model_name: str = "yolo26m-pose.pt"):
        self.model = YOLO(ASSETS_DIR / model_name)

    def run(self, image_bytes: bytes) -> bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model(frame)
        annotated = results[0].plot()

        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Failed to encode annotated image")

        return encoded.tobytes()


pose_service = PoseService()
