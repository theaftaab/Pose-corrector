"""
Quick visual test for the /api/v1/pose endpoint.

Usage:
    python tests/visualize_response.py [image_path] [--url URL]

Defaults to the bundled zidane.jpg and http://localhost:8000/api/v1/pose.
The annotated image is shown with matplotlib and also saved to /tmp/pose_result.jpg.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import requests


DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / ".venv/lib/python3.12/site-packages/ultralytics/assets/zidane.jpg"
)
DEFAULT_URL = "http://localhost:8000/api/v1/pose"


def call_api(image_path: Path, url: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"image": (image_path.name, f, "image/jpeg")})
    resp.raise_for_status()
    return resp.json()


def show(data: dict, original_path: Path):
    import cv2

    img_bytes = base64.b64decode(data["image_base64"])
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    original = mpimg.imread(str(original_path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    people = data.get("people", [])
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f"Annotated  |  {len(people)} person(s) detected")
    axes[1].axis("off")

    for person in people:
        pid = person["person_id"]
        print(f"\nPerson {pid + 1}  view={person['view']}  detected={person['person_detected']}")
        metrics = person.get("metrics", [])
        if metrics:
            print(f"  {'Region':<20} {'Value':>8} {'Unit':<8} {'Target':>8} {'Status':<12}")
            print("  " + "-" * 58)
            for m in metrics:
                print(
                    f"  {m['region']:<20} {m['value']:>8.1f} {m['unit']:<8} "
                    f"{str(m.get('target', '-')):>8} {m.get('status', '-'):<12}"
                )

    out_path = Path("/tmp/pose_result.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated image saved to {out_path}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize pose API response")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE), help="Path to input image")
    parser.add_argument("--url", default=DEFAULT_URL, help="API endpoint URL")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {image_path.name} to {args.url} ...")
    data = call_api(image_path, args.url)
    show(data, image_path)


if __name__ == "__main__":
    main()
