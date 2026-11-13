"""Training loss for the grid detector.

Positive predictors — the ones assigned a ground-truth animal — pull the
box and the class. Every predictor also learns objectness, with empty
cells weighted down so the few real animals are not drowned out by the
7×7×2 background slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ml.model import BOXES_PER_CELL, GRID


def detection_loss(
    prediction: torch.Tensor,
    boxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    *,
    coord_weight: float = 5.0,
    no_object_weight: float = 0.6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the mean loss and a breakdown for the training log.

    ``boxes`` are normalized ``cx, cy, w, h`` in 0..1. ``labels`` are class
    indices. Both lists are one entry per image in the batch.
    """
    if prediction.ndim != 5:
        raise ValueError(f"expected N,S,S,B,C prediction, got {tuple(prediction.shape)}")

    batch, grid_y, grid_x, predictors, _ = prediction.shape
    if (grid_y, grid_x, predictors) != (GRID, GRID, BOXES_PER_CELL):
        raise ValueError(
            f"this loss is written for a {GRID}×{GRID} grid with "
            f"{BOXES_PER_CELL} predictors"
        )

    device = prediction.device
    obj_target = torch.zeros(batch, GRID, GRID, BOXES_PER_CELL, device=device)
    box_target = torch.zeros(batch, GRID, GRID, BOXES_PER_CELL, 4, device=device)
    class_target = torch.zeros(
        batch, GRID, GRID, BOXES_PER_CELL, dtype=torch.long, device=device
    )
    positive = torch.zeros(
        batch, GRID, GRID, BOXES_PER_CELL, dtype=torch.bool, device=device
    )

    for image_index, (image_boxes, image_labels) in enumerate(zip(boxes, labels)):
        _assign_image(
            image_boxes,
            image_labels,
            obj_target[image_index],
            box_target[image_index],
            class_target[image_index],
            positive[image_index],
        )

    xy = torch.sigmoid(prediction[..., 0:2])
    wh = torch.sigmoid(prediction[..., 2:4])
    obj_logit = prediction[..., 4]
    class_logit = prediction[..., 5:]

    # Average the animal slots and the empty slots separately. A batch has
    # about a hundred empty predictors for every animal, and a plain sum
    # lets that background term drown the box and the class.
    if positive.any():
        loss_xy = ((xy - box_target[..., 0:2]) ** 2).sum(dim=-1)[positive].mean()
        loss_wh = ((wh - box_target[..., 2:4]) ** 2).sum(dim=-1)[positive].mean()
        loss_cls = F.cross_entropy(class_logit[positive], class_target[positive])
    else:
        loss_xy = prediction.new_zeros(())
        loss_wh = prediction.new_zeros(())
        loss_cls = prediction.new_zeros(())

    obj_bce = F.binary_cross_entropy_with_logits(obj_logit, obj_target, reduction="none")
    loss_obj = obj_bce[positive].mean() if positive.any() else prediction.new_zeros(())
    if (~positive).any():
        loss_obj = loss_obj + no_object_weight * obj_bce[~positive].mean()

    total = coord_weight * (loss_xy + loss_wh) + loss_obj + loss_cls
    parts = {
        "loss": float(total.detach()),
        "box": float((loss_xy + loss_wh).detach()),
        "object": float(loss_obj.detach()),
        "class": float(loss_cls.detach()),
    }
    return total, parts


def auxiliary_loss(logits: torch.Tensor, boxes: list[torch.Tensor], labels: list[torch.Tensor]):
    """Classify the largest animal in the frame. Empty frames are skipped."""
    chosen_logits = []
    chosen_labels = []
    for image_logits, image_boxes, image_labels in zip(logits, boxes, labels):
        if image_labels.numel() == 0:
            continue
        areas = image_boxes[:, 2] * image_boxes[:, 3]
        chosen_logits.append(image_logits)
        chosen_labels.append(image_labels[int(areas.argmax().item())])
    if not chosen_labels:
        return logits.new_zeros(())
    stacked_logits = torch.stack(chosen_logits)
    stacked_labels = torch.stack(chosen_labels)
    return F.cross_entropy(stacked_logits, stacked_labels)


def _assign_image(boxes, labels, obj_target, box_target, class_target, positive) -> None:
    if boxes.numel() == 0:
        return

    # Prefer larger animals when a cell can only keep two.
    areas = boxes[:, 2] * boxes[:, 3]
    order = torch.argsort(areas, descending=True)
    used = torch.zeros(GRID, GRID, dtype=torch.long)

    for index in order.tolist():
        cx, cy, width, height = boxes[index].tolist()
        cx = min(max(cx, 0.0), 1.0 - 1e-6)
        cy = min(max(cy, 0.0), 1.0 - 1e-6)
        cell_x = int(cx * GRID)
        cell_y = int(cy * GRID)
        slot = int(used[cell_y, cell_x].item())
        if slot >= BOXES_PER_CELL:
            continue
        used[cell_y, cell_x] = slot + 1

        obj_target[cell_y, cell_x, slot] = 1.0
        box_target[cell_y, cell_x, slot, 0] = cx * GRID - cell_x
        box_target[cell_y, cell_x, slot, 1] = cy * GRID - cell_y
        box_target[cell_y, cell_x, slot, 2] = min(max(width, 1.0 / 224), 1.0)
        box_target[cell_y, cell_x, slot, 3] = min(max(height, 1.0 / 224), 1.0)
        class_target[cell_y, cell_x, slot] = int(labels[index].item())
        positive[cell_y, cell_x, slot] = True
