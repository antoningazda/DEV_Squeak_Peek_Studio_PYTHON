"""
CNN detector training: Faster R-CNN over spectrogram tiles.

Companion to squeak_peek.ml.train (the Random Forest frame-classifier
detector) — this is the object-detection alternative, predicting a
bounding box (start/end time, frequency band) per call directly, the way
DeepSqueak's own detector works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from squeak_peek.cnn.dataset import USVBoxDataset, collate_fn
from squeak_peek.cnn.model import Backbone, build_fasterrcnn


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_cnn(
    wav_label_pairs: list[tuple[str | Path, str | Path]],
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    fcut_min: float = 40_000,
    fcut_max: float = 120_000,
    segment_length: int = 1024,
    overlap_factor: float = 0.5,
    negative_ratio: float = 1.0,
    backbone: Backbone = "mobilenet",
    pretrained_backbone: bool = True,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_fraction: float = 0.15,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a Faster R-CNN USV detector from (wav, label) file pairs.

    Returns a checkpoint dict: {"model_state_dict", "backbone",
    "tile_params", "training_info"} — see save_checkpoint()/load_checkpoint().
    """
    dataset = USVBoxDataset(
        wav_label_pairs,
        window_s=window_s, hop_s=hop_s,
        fcut_min=fcut_min, fcut_max=fcut_max,
        segment_length=segment_length, overlap_factor=overlap_factor,
        negative_ratio=negative_ratio, seed=seed,
    )

    n_val = max(1, round(len(dataset) * val_fraction)) if len(dataset) > 4 else 0
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = (
        random_split(dataset, [n_train, n_val], generator=generator) if n_val else (dataset, None)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dev = _pick_device(device)
    model = build_fasterrcnn(num_classes=2, backbone=backbone, pretrained_backbone=pretrained_backbone)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for images, targets in train_loader:
            images = [img.to(dev) for img in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
        epoch_losses.append(running_loss / max(n_batches, 1))

    n_val_detections = n_val_gt = 0
    if val_set is not None and len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for i in range(len(val_set)):
                image, target = val_set[i]
                pred = model([image.to(dev)])[0]
                n_val_detections += int((pred["scores"] > 0.5).sum())
                n_val_gt += int(target["boxes"].shape[0])

    return {
        "model_state_dict": model.state_dict(),
        "backbone": backbone,
        "tile_params": {
            "window_s": window_s,
            "fcut_min": fcut_min,
            "fcut_max": fcut_max,
            "segment_length": segment_length,
            "overlap_factor": overlap_factor,
        },
        "training_info": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_tiles_total": len(dataset),
            "n_tiles_train": n_train,
            "n_tiles_val": n_val,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "val_detections_at_0.5": n_val_detections,
            "val_ground_truth_boxes": n_val_gt,
        },
    }


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    """Load a checkpoint saved by save_checkpoint() and rebuild the model in eval() mode."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CNN model not found: {path}")

    dev = _pick_device(device)
    checkpoint = torch.load(path, map_location=dev, weights_only=False)

    model = build_fasterrcnn(num_classes=2, backbone=checkpoint["backbone"], pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()

    checkpoint["model"] = model
    checkpoint["device"] = dev
    return checkpoint
