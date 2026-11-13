"""Train the animal detector on Pascal VOC 2007 photos.

The best weights, an ONNX copy, and a short report are written under
``public/models`` so the web app can run the same network.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.data import AnimalImages, make_splits
from ml.decode import decode_predictions, match_recall
from ml.loss import auxiliary_loss, detection_loss
from ml.model import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, AnimalDetector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "VOCdevkit" / "VOC2007"
EXPORT_DIR = ROOT / "public" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Fieldmark animal detector")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--freeze-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=0, help="cap each split, for a smoke run")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(4)

    train_samples, val_samples = make_splits(args.data)
    if args.limit:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: max(8, args.limit // 5)]

    print(
        f"train images {len(train_samples)}   val images {len(val_samples)}   "
        f"classes {', '.join(CLASSES)}"
    )

    train_loader = DataLoader(
        AnimalImages(train_samples, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        AnimalImages(val_samples, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    model = AnimalDetector()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"parameters {parameter_count:,}")
    optimizer = None
    schedule = None

    best_f1 = -1.0
    best_record = None
    best_state = None
    history = []
    started = time.time()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = EXPORT_DIR / "animal-detector.pt"

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        if epoch == 1 or epoch == args.freeze_epochs + 1:
            model.set_backbone_frozen(frozen)
            learning_rate = args.lr if frozen else args.finetune_lr
            remaining = args.freeze_epochs if frozen else args.epochs - args.freeze_epochs
            optimizer = torch.optim.AdamW(
                (param for param in model.parameters() if param.requires_grad),
                lr=learning_rate,
                weight_decay=1e-4,
            )
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1)
            )
            print(
                f"{'frozen trunk' if frozen else 'fine-tuning'}  lr {learning_rate:g}",
                flush=True,
            )
        model.train()
        if frozen:
            model.set_backbone_frozen(True)
        running = 0.0
        part_sums = {"box": 0.0, "object": 0.0, "class": 0.0, "aux": 0.0}
        seen = 0
        epoch_started = time.time()
        for images, boxes, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, aux_logits = model.forward_train(images)
            loss, parts = detection_loss(prediction, boxes, labels)
            aux = auxiliary_loss(aux_logits, boxes, labels)
            loss = loss + aux
            parts = {**parts, "loss": float(loss.detach()), "aux": float(aux.detach())}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            assert optimizer is not None and schedule is not None
            optimizer.step()
            running += float(loss.detach()) * images.shape[0]
            for key in part_sums:
                part_sums[key] += parts[key] * images.shape[0]
            seen += images.shape[0]
        schedule.step()

        metrics = evaluate(model, val_loader, args.score_threshold)
        train_loss = running / max(seen, 1)
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        record = {
            "epoch": epoch,
            "trainLoss": round(train_loss, 4),
            "valLoss": round(metrics["loss"], 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "maxScore": round(metrics["maxScore"], 4),
            "seconds": round(time.time() - epoch_started, 1),
        }
        history.append(record)
        if f1 > best_f1:
            best_f1 = f1
            best_record = record
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "classes": list(CLASSES)}, weights_path)
        print(
            f"epoch {epoch:02d}  train {train_loss:.3f}  "
            f"box {part_sums['box'] / max(seen, 1):.3f}  "
            f"obj {part_sums['object'] / max(seen, 1):.3f}  "
            f"cls {part_sums['class'] / max(seen, 1):.3f}  "
            f"aux {part_sums['aux'] / max(seen, 1):.3f}  "
            f"val {metrics['loss']:.3f}  recall {recall:.3f}  "
            f"precision {precision:.3f}  f1 {f1:.3f}  max {metrics['maxScore']:.3f}  "
            f"{record['seconds']:.0f}s",
            flush=True,
        )
    if best_state is None:
        raise SystemExit("training produced no weights")

    model.load_state_dict(best_state)
    per_class = per_class_recall(model, val_loader, args.score_threshold)
    spec = {
        "classes": list(CLASSES),
        "imageSize": IMAGE_SIZE,
        "grid": model.grid,
        "boxesPerCell": model.boxes_per_cell,
        "fill": 114,
        "channels": 5 + len(CLASSES),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    report = {
        "dataset": "Pascal VOC 2007 trainval, animal boxes only",
        "classes": list(CLASSES),
        "imageSize": IMAGE_SIZE,
        "grid": model.grid,
        "boxesPerCell": model.boxes_per_cell,
        "parameters": parameter_count,
        "epochs": args.epochs,
        "bestRecall": None if best_record is None else best_record["recall"],
        "bestPrecision": None if best_record is None else best_record["precision"],
        "bestF1": round(best_f1, 4),
        "perClassRecall": per_class,
        "history": history,
        "trainedSeconds": round(time.time() - started, 1),
        "layers": model.describe(),
    }
    (EXPORT_DIR / "spec.json").write_text(json.dumps(spec, indent=2))
    torch.save({"state_dict": best_state, "classes": list(CLASSES)}, weights_path)
    export_onnx(model, EXPORT_DIR / "animal-detector.onnx")
    (EXPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {weights_path}")
    print(json.dumps({"bestRecall": report["bestRecall"], "perClassRecall": per_class}, indent=2))


def evaluate(model: AnimalDetector, loader: DataLoader, score_threshold: float) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    images_seen = 0
    matched = 0
    ground_truth = 0
    predicted = 0
    max_score = 0.0
    with torch.no_grad():
        for images, boxes, labels in loader:
            prediction = model(images)
            loss, _parts = detection_loss(prediction, boxes, labels)
            loss_total += float(loss) * images.shape[0]
            images_seen += images.shape[0]
            obj = torch.sigmoid(prediction[..., 4])
            class_score = torch.softmax(prediction[..., 5:], dim=-1).max(dim=-1).values
            max_score = max(max_score, float((obj * class_score).max()))
            decoded = decode_predictions(prediction, score_threshold=score_threshold)
            for image_predictions, image_boxes, image_labels in zip(decoded, boxes, labels):
                hit, count, preds = match_recall(image_predictions, image_boxes, image_labels)
                matched += hit
                ground_truth += count
                predicted += preds
    return {
        "loss": loss_total / max(images_seen, 1),
        "recall": matched / ground_truth if ground_truth else 0.0,
        "precision": matched / predicted if predicted else 0.0,
        "maxScore": max_score,
    }


def per_class_recall(model: AnimalDetector, loader: DataLoader, score_threshold: float) -> dict[str, float]:
    model.eval()
    matched = [0] * len(CLASSES)
    total = [0] * len(CLASSES)
    with torch.no_grad():
        for images, boxes, labels in loader:
            decoded = decode_predictions(model(images), score_threshold=score_threshold)
            for image_predictions, image_boxes, image_labels in zip(decoded, boxes, labels):
                for class_index, name in enumerate(CLASSES):
                    keep = image_labels == class_index
                    hit, count, _preds = match_recall(
                        [item for item in image_predictions if item["class_id"] == class_index],
                        image_boxes[keep],
                        image_labels[keep],
                    )
                    matched[class_index] += hit
                    total[class_index] += count
                    _ = name
    return {
        name: round(matched[index] / total[index], 4) if total[index] else 0.0
        for index, name in enumerate(CLASSES)
    }


def export_onnx(model: AnimalDetector, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["image"],
        output_names=["prediction"],
        opset_version=17,
        dynamo=False,
    )


def _collate(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels


if __name__ == "__main__":
    main()
