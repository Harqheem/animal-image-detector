"""Turn the network's grid into boxes.

This is the decode used at training time and by the command-line checker.
The browser copy in ``src/lib/decode.ts`` follows the same arithmetic.
"""

from __future__ import annotations

import torch

from ml.model import BOXES_PER_CELL, CLASSES, GRID


def decode_predictions(
    prediction: torch.Tensor,
    score_threshold: float = 0.3,
    iou_threshold: float = 0.3,
) -> list[list[dict]]:
    """Decode a batch.

    Each detection is ``class_id``, ``name``, ``score``, and ``x1,y1,x2,y2``
    normalized to the letterboxed network input, origin at the top left.
    """
    if prediction.ndim == 4:
        prediction = prediction.unsqueeze(0)
    prediction = prediction.detach().float().cpu()

    decoded: list[list[dict]] = []
    for image in prediction:
        decoded.append(_decode_image(image, score_threshold, iou_threshold))
    return decoded


def _decode_image(image: torch.Tensor, score_threshold: float, iou_threshold: float) -> list[dict]:
    xy = torch.sigmoid(image[..., 0:2])
    wh = torch.sigmoid(image[..., 2:4])
    obj = torch.sigmoid(image[..., 4])
    class_prob = torch.softmax(image[..., 5:], dim=-1)
    class_score, class_id = class_prob.max(dim=-1)
    score = obj * class_score

    candidates: list[dict] = []
    grid_y, grid_x, predictors = score.shape
    if (grid_y, grid_x, predictors) != (GRID, GRID, BOXES_PER_CELL):
        raise ValueError(f"unexpected grid {tuple(score.shape)}")

    ys, xs, slots = torch.where(score >= score_threshold)
    for cell_y, cell_x, slot in zip(ys.tolist(), xs.tolist(), slots.tolist()):
        center_x = (xy[cell_y, cell_x, slot, 0].item() + cell_x) / GRID
        center_y = (xy[cell_y, cell_x, slot, 1].item() + cell_y) / GRID
        width = wh[cell_y, cell_x, slot, 0].item()
        height = wh[cell_y, cell_x, slot, 1].item()
        x1 = max(0.0, center_x - width / 2)
        y1 = max(0.0, center_y - height / 2)
        x2 = min(1.0, center_x + width / 2)
        y2 = min(1.0, center_y + height / 2)
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            continue
        kind = int(class_id[cell_y, cell_x, slot].item())
        candidates.append(
            {
                "class_id": kind,
                "name": CLASSES[kind],
                "score": float(score[cell_y, cell_x, slot].item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return _nms(candidates, iou_threshold)


def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Keep the stronger box when two marks land on the same animal."""
    ordered = sorted(detections, key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [item for item in ordered if _iou(best, item) < iou_threshold]
    return kept


def _iou(a: dict, b: dict) -> float:
    left = max(a["x1"], b["x1"])
    top = max(a["y1"], b["y1"])
    right = min(a["x2"], b["x2"])
    bottom = min(a["y2"], b["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_recall(
    predictions: list[dict],
    boxes: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Return matched ground truth, ground-truth count, and prediction count."""
    if boxes.numel() == 0:
        return 0, 0, len(predictions)

    remaining = predictions[:]
    matched = 0
    for box, label in zip(boxes.tolist(), labels.tolist()):
        cx, cy, width, height = box
        truth = {
            "x1": cx - width / 2,
            "y1": cy - height / 2,
            "x2": cx + width / 2,
            "y2": cy + height / 2,
        }
        best_index = -1
        best_iou = iou_threshold
        for index, prediction in enumerate(remaining):
            if prediction["class_id"] != int(label):
                continue
            overlap = _iou(prediction, truth)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0:
            matched += 1
            remaining.pop(best_index)
    return matched, int(labels.shape[0]), len(predictions)
