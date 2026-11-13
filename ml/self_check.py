"""Prove the loss can teach the network before the real photo run.

Four plain canvases, each with one colored rectangle, are memorized.
If recall on those four is short of perfect, the grid math is wrong and
a long training run would be wasted.
"""

from __future__ import annotations

import torch

from ml.decode import decode_predictions, match_recall
from ml.loss import detection_loss
from ml.model import IMAGE_SIZE, AnimalDetector


def _scene(color: tuple[float, float, float], box: tuple[int, int, int, int], label: int):
    image = torch.full((3, IMAGE_SIZE, IMAGE_SIZE), 114 / 255)
    x1, y1, x2, y2 = box
    tone = torch.tensor(color).view(3, 1, 1)
    image[:, y1:y2, x1:x2] = tone
    cx = ((x1 + x2) / 2) / IMAGE_SIZE
    cy = ((y1 + y2) / 2) / IMAGE_SIZE
    width = (x2 - x1) / IMAGE_SIZE
    height = (y2 - y1) / IMAGE_SIZE
    boxes = torch.tensor([[cx, cy, width, height]], dtype=torch.float32)
    labels = torch.tensor([label], dtype=torch.long)
    return image, boxes, labels


def main() -> None:
    torch.manual_seed(0)
    scenes = [
        _scene((0.85, 0.25, 0.2), (30, 40, 120, 150), 1),  # cat
        _scene((0.2, 0.45, 0.85), (90, 20, 200, 180), 0),  # bird
        _scene((0.25, 0.65, 0.35), (20, 100, 110, 210), 2),  # dog
        _scene((0.75, 0.55, 0.2), (100, 80, 210, 200), 3),  # horse
    ]
    images = torch.stack([scene[0] for scene in scenes])
    boxes = [scene[1] for scene in scenes]
    labels = [scene[2] for scene in scenes]

    model = AnimalDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    model.train()
    last_loss = None
    for step in range(1, 161):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = detection_loss(model(images), boxes, labels)
        loss.backward()
        optimizer.step()
        last_loss = parts["loss"]
        if step % 40 == 0:
            print(f"step {step:3d}  loss {parts['loss']:.4f}  box {parts['box']:.4f}  class {parts['class']:.4f}")

    model.eval()
    with torch.no_grad():
        predictions = decode_predictions(model(images), score_threshold=0.25)
    matched = 0
    total = 0
    for prediction, box, label in zip(predictions, boxes, labels):
        hit, count, _ = match_recall(prediction, box, label)
        matched += hit
        total += count
        print(
            "scene",
            label.item(),
            "->",
            [(item["name"], round(item["score"], 3)) for item in prediction],
        )
    recall = matched / total
    print(f"synthetic recall {matched}/{total} = {recall:.2f}  final loss {last_loss:.4f}")
    if matched != total:
        raise SystemExit("the network did not learn the synthetic animals")


if __name__ == "__main__":
    main()
