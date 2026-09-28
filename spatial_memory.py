"""Turn one image into image-relative object-location records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class Detection:
    """A detector result in source-image pixel coordinates."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_observation(
    image_path: Path,
    image_width: int,
    image_height: int,
    detections: Iterable[Detection],
    model_name: str,
) -> dict:
    """Create a JSON-serializable observation with normalized 2D locations."""
    observation_id = str(uuid.uuid4())
    objects = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        objects.append(
            {
                "object_id": str(uuid.uuid4()),
                "label": detection.label,
                "confidence": round(float(detection.confidence), 6),
                "location": {
                    "coordinate_frame": "image",
                    "bbox_xyxy_px": [round(v, 2) for v in (x1, y1, x2, y2)],
                    "bbox_xyxy_normalized": [
                        round(x1 / image_width, 6),
                        round(y1 / image_height, 6),
                        round(x2 / image_width, 6),
                        round(y2 / image_height, 6),
                    ],
                    "center_normalized": [
                        round(((x1 + x2) / 2) / image_width, 6),
                        round(((y1 + y2) / 2) / image_height, 6),
                    ],
                },
            }
        )

    return {
        "observation_id": observation_id,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "source_image": image_path.name,
        "source_sha256": file_sha256(image_path),
        "image_size_px": {"width": image_width, "height": image_height},
        "detector": model_name,
        "objects": objects,
    }


def detect_objects(image_path: Path, model_name: str, confidence: float) -> list[Detection]:
    """Run Ultralytics YOLO and return detections in original-image pixels."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Object detection needs Ultralytics. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from error

    model = YOLO(model_name)
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                Detection(
                    label=str(result.names[class_id]),
                    confidence=float(box.conf.item()),
                    bbox_xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
                )
            )
    return detections


def append_observation(observation: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(observation, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect objects in an image and append image-relative locations to JSONL."
    )
    parser.add_argument("image", type=Path, help="Path to a local image")
    parser.add_argument("--output", type=Path, default=Path("data/observations.jsonl"))
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Ultralytics model name or weights path"
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist or is not a file: {args.image}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")

    try:
        with Image.open(args.image) as image:
            image_width, image_height = image.size
            image.verify()
        detections = detect_objects(args.image, args.model, args.confidence)
        observation = build_observation(
            args.image, image_width, image_height, detections, args.model
        )
        append_observation(observation, args.output)
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(observation['objects'])} object(s) from {args.image.name} "
        f"to {args.output} (observation {observation['observation_id']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
