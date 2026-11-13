"""Pascal VOC 2007 loader for the six animal classes this network learns.

Images that contain a bird, cat, dog, horse, sheep, or cow keep those boxes
and ignore everything else (people, cars, chairs). A smaller set of photos
with none of those animals is kept so the network can learn an empty grid.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from ml.model import CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
LETTERBOX_FILL = 114


@dataclass
class Sample:
    path: Path
    boxes: list[tuple[float, float, float, float]]
    labels: list[int]


def letterbox_geometry(width: int, height: int, size: int = IMAGE_SIZE):
    """Scale and padding shared with the browser preprocessor.

    Both sides use floor division so a photo is resized the same way in
    Python and in the page that runs the exported network.
    """
    scale = size / max(width, height)
    resized_w = max(1, min(size, int(width * scale)))
    resized_h = max(1, min(size, int(height * scale)))
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    return scale, resized_w, resized_h, pad_x, pad_y


def discover_samples(voc_root: Path) -> tuple[list[Sample], list[Sample]]:
    jpeg_dir = voc_root / "JPEGImages"
    anno_dir = voc_root / "Annotations"
    if not jpeg_dir.is_dir() or not anno_dir.is_dir():
        raise FileNotFoundError(
            f"expected VOC JPEGImages and Annotations under {voc_root}"
        )

    animals: list[Sample] = []
    background: list[Sample] = []
    for annotation_path in sorted(anno_dir.glob("*.xml")):
        boxes, labels = _read_animals(annotation_path)
        image_path = jpeg_dir / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            continue
        sample = Sample(image_path, boxes, labels)
        if labels:
            animals.append(sample)
        else:
            background.append(sample)
    return animals, background


def make_splits(
    voc_root: Path,
    val_fraction: float = 0.15,
    background_ratio: float = 0.35,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    animals, background = discover_samples(voc_root)
    rng = random.Random(seed)
    rng.shuffle(animals)
    rng.shuffle(background)

    val_count = max(1, int(len(animals) * val_fraction))
    val_animals = animals[:val_count]
    train_animals = animals[val_count:]

    def take_background(pool: list[Sample], count: int) -> list[Sample]:
        if not pool or count <= 0:
            return []
        if count >= len(pool):
            return pool
        return pool[:count]

    # Background photos are drawn from the tail so validation animals and
    # validation backgrounds do not share a shuffle position with training.
    train_background = take_background(
        background[val_count:], int(len(train_animals) * background_ratio)
    )
    val_background = take_background(
        background[:val_count], int(len(val_animals) * background_ratio)
    )
    train = train_animals + train_background
    val = val_animals + val_background
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class AnimalImages(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.samples = samples
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        boxes = [list(box) for box in sample.boxes]
        labels = list(sample.labels)

        if self.train:
            image, boxes = _zoom(image, boxes)
        if self.train and boxes and random.random() < 0.5:
            image, boxes = _flip(image, boxes)
        if self.train:
            image = _jitter_color(image)

        tensor, boxes = _letterbox(image, boxes)
        return (
            tensor,
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long),
        )


def _read_animals(path: Path) -> tuple[list[tuple[float, float, float, float]], list[int]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        return [], []
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 1 or height <= 1:
        return [], []

    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_TO_INDEX:
            continue
        if (obj.findtext("difficult") or "0").strip() == "1":
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        xmax = min(xmax, width)
        ymax = min(ymax, height)
        xmin = max(0.0, min(xmin, xmax - 1))
        ymin = max(0.0, min(ymin, ymax - 1))
        cx = ((xmin + xmax) / 2) / width
        cy = ((ymin + ymax) / 2) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        if bw < 0.01 or bh < 0.01:
            continue
        boxes.append((cx, cy, bw, bh))
        labels.append(CLASS_TO_INDEX[name])
    return boxes, labels


def _zoom(image: Image.Image, boxes: list[list[float]]):
    """Crop inward while keeping every labeled animal inside the window."""
    if random.random() > 0.55:
        return image, boxes
    width, height = image.size
    zoom = random.uniform(1.05, 1.45)
    crop_w = width / zoom
    crop_h = height / zoom
    if boxes:
        min_x = min(box[0] - box[2] / 2 for box in boxes) * width
        max_x = max(box[0] + box[2] / 2 for box in boxes) * width
        min_y = min(box[1] - box[3] / 2 for box in boxes) * height
        max_y = max(box[1] + box[3] / 2 for box in boxes) * height
        if max_x - min_x > crop_w or max_y - min_y > crop_h:
            return image, boxes
        left_min = max(0.0, max_x - crop_w)
        left_max = min(min_x, width - crop_w)
        top_min = max(0.0, max_y - crop_h)
        top_max = min(min_y, height - crop_h)
        if left_max < left_min or top_max < top_min:
            return image, boxes
        left = random.uniform(left_min, left_max)
        top = random.uniform(top_min, top_max)
    else:
        left = random.uniform(0.0, max(0.0, width - crop_w))
        top = random.uniform(0.0, max(0.0, height - crop_h))

    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    moved = [
        [
            (box[0] * width - left) / crop_w,
            (box[1] * height - top) / crop_h,
            box[2] * width / crop_w,
            box[3] * height / crop_h,
        ]
        for box in boxes
    ]
    return cropped, moved


def _flip(image: Image.Image, boxes: list[list[float]]):
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    for box in boxes:
        box[0] = 1.0 - box[0]
    return image, boxes


def _jitter_color(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
    return image


def _letterbox(image: Image.Image, boxes: list[list[float]]):
    width, height = image.size
    scale, resized_w, resized_h, pad_x, pad_y = letterbox_geometry(width, height)
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (LETTERBOX_FILL, LETTERBOX_FILL, LETTERBOX_FILL))
    canvas.paste(resized, (pad_x, pad_y))

    translated: list[list[float]] = []
    for cx, cy, bw, bh in boxes:
        abs_cx = cx * width * scale + pad_x
        abs_cy = cy * height * scale + pad_y
        abs_w = bw * width * scale
        abs_h = bh * height * scale
        translated.append(
            [
                abs_cx / IMAGE_SIZE,
                abs_cy / IMAGE_SIZE,
                abs_w / IMAGE_SIZE,
                abs_h / IMAGE_SIZE,
            ]
        )

    array = torch.from_numpy(np.asarray(canvas, dtype=np.float32))
    tensor = array.permute(2, 0, 1).contiguous() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, translated
