"""
Faster R-CNN model builder for the CNN USV detector.

Default backbone is MobileNetV3 (fasterrcnn_mobilenet_v3_large_320_fpn):
much cheaper to train/run on a CPU-only researcher laptop than the
ResNet50-FPN backbone DeepSqueak itself used, at some accuracy cost.
`backbone="resnet50"` opts into the heavier, more accurate network for
users with a GPU.
"""

from __future__ import annotations

from typing import Literal

Backbone = Literal["mobilenet", "resnet50"]


def build_fasterrcnn(
    num_classes: int = 2,
    *,
    backbone: Backbone = "mobilenet",
    pretrained_backbone: bool = True,
):
    """
    Build a torchvision Faster R-CNN with a random detection head sized
    for `num_classes` (background + USV, by default). The backbone is
    ImageNet-pretrained by default (transfer learning matters a lot given
    how little labeled USV data exists publicly).
    """
    from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )

    if backbone == "mobilenet":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, num_classes=num_classes, weights_backbone=weights_backbone,
        )
    if backbone == "resnet50":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        return fasterrcnn_resnet50_fpn_v2(
            weights=None, num_classes=num_classes, weights_backbone=weights_backbone,
        )
    raise ValueError(f"build_fasterrcnn: unknown backbone {backbone!r}, expected 'mobilenet' or 'resnet50'.")
