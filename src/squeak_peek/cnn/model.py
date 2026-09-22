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
    anchor_sizes: tuple[tuple[int, ...], ...] | None = None,
    aspect_ratios: tuple[tuple[float, ...], ...] | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
):
    """
    Build a torchvision Faster R-CNN with a random detection head sized
    for `num_classes` (background + USV, by default). The backbone is
    ImageNet-pretrained by default (transfer learning matters a lot given
    how little labeled USV data exists publicly).

    `anchor_sizes`/`aspect_ratios` override torchvision's COCO anchors
    (sizes 32-512, ratios 0.5-2). USV calls on a spectrogram are small and
    elongated — a median USVSEG mouse call is ~27 ms x 8 kHz, which lands
    well under the smallest COCO anchor — so USV models want smaller sizes
    and taller aspect ratios. `min_size`/`max_size` override the input
    rescaling, which sets how many pixels a call ends up spanning.
    """
    from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )

    if backbone == "mobilenet":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, num_classes=num_classes, weights_backbone=weights_backbone,
        )
    elif backbone == "resnet50":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None, num_classes=num_classes, weights_backbone=weights_backbone,
        )
    else:
        raise ValueError(f"build_fasterrcnn: unknown backbone {backbone!r}, expected 'mobilenet' or 'resnet50'.")

    if anchor_sizes is not None or aspect_ratios is not None:
        _retarget_anchors(model, anchor_sizes, aspect_ratios)
    if min_size is not None:
        model.transform.min_size = (min_size,)
    if max_size is not None:
        model.transform.max_size = max_size
    return model


def _retarget_anchors(model, anchor_sizes, aspect_ratios) -> None:
    """Swap in a new anchor generator, resizing the RPN head to match its anchor count."""
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.rpn import RPNHead

    current = model.rpn.anchor_generator
    sizes = tuple(anchor_sizes) if anchor_sizes is not None else current.sizes
    ratios = tuple(aspect_ratios) if aspect_ratios is not None else current.aspect_ratios
    if len(ratios) != len(sizes):
        ratios = tuple(ratios[0] for _ in sizes)

    generator = AnchorGenerator(sizes, ratios)
    model.rpn.anchor_generator = generator
    model.rpn.head = RPNHead(
        model.backbone.out_channels, generator.num_anchors_per_location()[0]
    )
