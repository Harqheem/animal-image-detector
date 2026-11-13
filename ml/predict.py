"""Run the trained detector on one image from the command line.

Prints one JSON object: the animal boxes, in reading order of confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from ml.data import _letterbox
from ml.decode import decode_predictions
from ml.model import AnimalDetector

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "public" / "models" / "animal-detector.pt"


def load_model(weights: Path = WEIGHTS) -> AnimalDetector:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = AnimalDetector()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_image(model: AnimalDetector, path: Path, score_threshold: float = 0.3) -> list[dict]:
    image = Image.open(path).convert("RGB")
    tensor, _boxes = _letterbox(image, [])
    with torch.no_grad():
        prediction = model(tensor.unsqueeze(0))
    detections = decode_predictions(prediction, score_threshold=score_threshold)[0]
    for detection in detections:
        for key in ("score", "x1", "y1", "x2", "y2"):
            detection[key] = round(detection[key], 4)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect animals in one image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    model = load_model(args.weights)
    print(json.dumps(predict_image(model, args.image, args.threshold), indent=2))


if __name__ == "__main__":
    main()
