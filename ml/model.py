"""Convolutional animal detector.

A 224×224 RGB image goes through a ResNet-18 trunk and comes out as a 7×7
grid. The trunk starts from ImageNet weights. The detection head on top of
it is trained in this project. Every cell holds two predictors. Each
predictor scores one box: where it sits inside the cell, how large it is,
whether an animal is there, and which animal it is.

The raw tensor returned by ``forward`` has shape

    (batch, 7, 7, 2, 5 + num_classes)

and is laid out as

    [..., 0]  center x inside the cell, before the sigmoid
    [..., 1]  center y inside the cell, before the sigmoid
    [..., 2]  width as a fraction of the image, before the sigmoid
    [..., 3]  height as a fraction of the image, before the sigmoid
    [..., 4]  objectness logit
    [..., 5:] class logits
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGE_SIZE = 224
GRID = 7
BOXES_PER_CELL = 2
CLASSES = ("bird", "cat", "dog", "horse", "sheep", "cow")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AnimalDetector(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.grid = GRID
        self.boxes_per_cell = BOXES_PER_CELL

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        trunk = resnet18(weights=weights)
        # 224 → 112 → 56 → 28 → 14 → 7, ending at 512 channels.
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

        self.head = nn.Conv2d(
            512,
            BOXES_PER_CELL * (5 + num_classes),
            kernel_size=1,
        )
        # Used only while training. It is not part of the exported graph.
        self.auxiliary = nn.Linear(512, num_classes)
        self._init_head()

    def backbone(self, images):
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return features

    def set_backbone_frozen(self, frozen: bool) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            module.requires_grad_(not frozen)
            module.eval() if frozen else module.train()

    def _init_head(self) -> None:
        nn.init.zeros_(self.head.weight)
        bias = self.head.bias
        assert bias is not None
        nn.init.zeros_(bias)
        span = 5 + self.num_classes
        with torch.no_grad():
            for box in range(BOXES_PER_CELL):
                base = box * span
                # A fresh box covers about a quarter of the frame.
                bias[base + 2] = -1.1
                bias[base + 3] = -1.1
                # Empty cells should start quiet.
                bias[base + 4] = -2.5

    def forward(self, images):
        return self._grid(self.backbone(images))

    def forward_train(self, images):
        features = self.backbone(images)
        pooled = features.mean(dim=(2, 3))
        return self._grid(features), self.auxiliary(pooled)

    def _grid(self, features):
        raw = self.head(features)
        batch = raw.shape[0]
        span = 5 + self.num_classes
        raw = raw.view(batch, self.boxes_per_cell, span, self.grid, self.grid)
        return raw.permute(0, 3, 4, 1, 2).contiguous()

    def describe(self) -> list[dict[str, int | str]]:
        """Layer list shown in the app and saved next to the exported weights."""
        rows: list[dict[str, int | str]] = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                rows.append(
                    {
                        "name": name,
                        "kind": "conv",
                        "inChannels": module.in_channels,
                        "outChannels": module.out_channels,
                        "kernel": module.kernel_size[0],
                        "stride": module.stride[0],
                    }
                )
            elif isinstance(module, nn.BatchNorm2d):
                rows.append(
                    {
                        "name": name,
                        "kind": "batchnorm",
                        "channels": module.num_features,
                    }
                )
            elif isinstance(module, nn.SiLU):
                rows.append({"name": name, "kind": "silu"})
        return rows
