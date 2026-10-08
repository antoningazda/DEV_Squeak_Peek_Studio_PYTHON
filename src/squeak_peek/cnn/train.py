"""
CNN detector training: Faster R-CNN over spectrogram tiles.

Companion to squeak_peek.ml.train (the Random Forest frame-classifier
detector) — this is the object-detection alternative, predicting a
bounding box (start/end time, frequency band) per call directly, the way
DeepSqueak's own detector works.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

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
    anchor_sizes: tuple[tuple[int, ...], ...] | None = None,
    aspect_ratios: tuple[tuple[float, ...], ...] | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_fraction: float = 0.15,
    device: str | None = None,
    seed: int = 42,
    progress: bool = False,
    progress_callback: Callable[[float | None, str], None] | None = None,
) -> dict[str, Any]:
    """
    Train a Faster R-CNN USV detector from (wav, label) file pairs.

    ``progress_callback(fraction, message)`` is called per batch; GUI callers
    may raise from it to cancel.

    Returns a checkpoint dict: {"model_state_dict", "backbone",
    "tile_params", "training_info"} — see save_checkpoint()/load_checkpoint().
    """
    if progress_callback is not None:
        progress_callback(None, "Loading recordings and tiling spectrograms…")
    dataset = USVBoxDataset(
        wav_label_pairs,
        window_s=window_s, hop_s=hop_s,
        fcut_min=fcut_min, fcut_max=fcut_max,
        segment_length=segment_length, overlap_factor=overlap_factor,
        negative_ratio=negative_ratio, seed=seed,
    )

    train_idx, val_idx = dataset.time_split_indices(val_fraction)
    if not train_idx:
        raise ValueError(
            f"train_cnn: val_fraction={val_fraction} left no training tiles. "
            "Use a smaller val_fraction or longer recordings."
        )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if val_idx else None
    n_train, n_val = len(train_idx), len(val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dev = _pick_device(device)
    model_params = {
        "anchor_sizes": anchor_sizes,
        "aspect_ratios": aspect_ratios,
        "min_size": min_size,
        "max_size": max_size,
    }
    model = build_fasterrcnn(
        num_classes=2, backbone=backbone, pretrained_backbone=pretrained_backbone, **model_params
    )
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    epoch_losses: list[float] = []
    n_skipped = 0
    n_steps = max(epochs * len(train_loader), 1)
    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for step, (images, targets) in enumerate(train_loader):
            if progress_callback is not None:
                last = f" · last epoch loss {epoch_losses[-1]:.4f}" if epoch_losses else ""
                progress_callback(
                    (epoch * len(train_loader) + step) / n_steps,
                    f"Epoch {epoch + 1}/{epochs} · batch {step + 1}/{len(train_loader)}{last}",
                )
            images = [img.to(dev) for img in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            # On a batch of entirely call-free tiles a trained RPN proposes no
            # regions, and torchvision's ROI-head loss divides by a zero
            # proposal count -> NaN in loss_classifier/loss_box_reg. Those
            # terms carry no gradient (empty-tensor reduction), but summing
            # them would make the reported loss NaN and hide the real curve.
            # The RPN terms in the same batch are finite and still useful.
            finite = [v for v in loss_dict.values() if torch.isfinite(v)]
            if len(finite) < len(loss_dict):
                n_skipped += 1
            if not finite:
                continue
            loss = sum(finite)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
        epoch_losses.append(running_loss / max(n_batches, 1))
        if progress:
            print(
                f"  epoch {epoch + 1}/{epochs}  loss {epoch_losses[-1]:.4f}"
                + (f"  ({n_skipped} batches had no proposals)" if n_skipped else ""),
                flush=True,
            )

    n_val_detections = n_val_gt = 0
    if val_set is not None and len(val_set) > 0:
        if progress_callback is not None:
            progress_callback(None, f"Validating on {len(val_set)} held-out tiles…")
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
        "model_params": model_params,
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
            "n_batches_without_proposals": n_skipped,
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

    model = build_fasterrcnn(
        num_classes=2, backbone=checkpoint["backbone"], pretrained_backbone=False,
        **checkpoint.get("model_params", {}),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()

    checkpoint["model"] = model
    checkpoint["device"] = dev
    return checkpoint
