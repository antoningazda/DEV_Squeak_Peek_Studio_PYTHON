"""
torch Dataset that tiles (wav, label) pairs into fixed-length spectrogram
images with Faster R-CNN box targets.

Each Label's (start_time, end_time, start_frequency, end_frequency) —
already the exact shape of a bounding box — is mapped into the tile's
pixel space. Reuses squeak_peek.labels.io.import_labels, so any label
file this repo already produces (GUI annotations, PSD/BSCD/RBD/ML
detector output, or squeak_peek.cnn.convert_usvseg output) is a valid
CNN training source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from squeak_peek.audio.io import load_wav
from squeak_peek.cnn.spectrogram_image import freq_to_row, signal_to_image, time_to_col
from squeak_peek.labels.io import import_labels
from squeak_peek.labels.model import Label


class USVBoxDataset(Dataset):
    """Windows one or more (wav, label) pairs into Faster R-CNN training tiles."""

    def __init__(
        self,
        wav_label_pairs: list[tuple[str | Path, str | Path]],
        *,
        window_s: float = 1.0,
        hop_s: float = 0.5,
        fcut_min: float = 40_000,
        fcut_max: float = 120_000,
        segment_length: int = 1024,
        overlap_factor: float = 0.5,
        negative_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not wav_label_pairs:
            raise ValueError("USVBoxDataset: no (wav, label) pairs given.")

        self.window_s = window_s
        self.fcut_min = fcut_min
        self.fcut_max = fcut_max
        self.segment_length = segment_length
        self.overlap_factor = overlap_factor

        self._signals: list[np.ndarray] = []
        self._fs: list[int] = []
        self._labels: list[list[Label]] = []

        positive_tiles: list[tuple[int, int]] = []
        negative_tiles: list[tuple[int, int]] = []

        for pair_idx, (wav_path, label_path) in enumerate(wav_label_pairs):
            signal, fs = load_wav(wav_path)
            # Calls rejected in Label Edit are false detections, not USVs.
            labels = [lbl for lbl in import_labels(label_path, fs) if lbl.detection_state != "Rejected"]
            self._signals.append(signal)
            self._fs.append(fs)
            self._labels.append(labels)

            window_len = round(window_s * fs)
            hop_len = round(hop_s * fs)
            if window_len <= 0 or hop_len <= 0 or len(signal) < window_len:
                continue

            for start in range(0, len(signal) - window_len + 1, hop_len):
                t0, t1 = start / fs, (start + window_len) / fs
                has_call = any(lbl.end_time > t0 and lbl.start_time < t1 for lbl in labels)
                (positive_tiles if has_call else negative_tiles).append((pair_idx, start))

        if not positive_tiles:
            raise ValueError(
                "USVBoxDataset: no tiles contain a labeled call. Check that "
                "window_s/hop_s and the label files actually line up with the WAVs."
            )

        rng = np.random.default_rng(seed)
        n_neg = min(len(negative_tiles), round(len(positive_tiles) * negative_ratio))
        if 0 < n_neg < len(negative_tiles):
            idx = rng.choice(len(negative_tiles), size=n_neg, replace=False)
            negative_tiles = [negative_tiles[i] for i in idx]
        elif n_neg == 0:
            negative_tiles = []

        self._tiles = positive_tiles + negative_tiles

    def __len__(self) -> int:
        return len(self._tiles)

    def time_split_indices(self, val_fraction: float) -> tuple[list[int], list[int]]:
        """
        Split tile indices by time within each recording, reserving the last
        `val_fraction` of every recording for validation.

        Tiles overlap (hop_s < window_s), so a random split would put tiles
        sharing the same audio on both sides and report an optimistic
        validation score. Tiles straddling the cut are dropped.
        """
        train_idx: list[int] = []
        val_idx: list[int] = []
        for i, (pair_idx, start) in enumerate(self._tiles):
            fs = self._fs[pair_idx]
            window_len = round(self.window_s * fs)
            cutoff = len(self._signals[pair_idx]) * (1.0 - val_fraction)
            if start >= cutoff:
                val_idx.append(i)
            elif start + window_len <= cutoff:
                train_idx.append(i)
        return train_idx, val_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pair_idx, start = self._tiles[idx]
        signal = self._signals[pair_idx]
        fs = self._fs[pair_idx]
        window_len = round(self.window_s * fs)
        end = start + window_len

        image, freqs, times = signal_to_image(
            signal[start:end], fs, self.fcut_min, self.fcut_max,
            segment_length=self.segment_length, overlap_factor=self.overlap_factor,
        )
        t0 = start / fs

        boxes: list[list[float]] = []
        for lbl in self._labels[pair_idx]:
            if lbl.end_time <= t0 or lbl.start_time >= t0 + self.window_s:
                continue
            x0 = time_to_col(times, max(0.0, lbl.start_time - t0))
            x1 = time_to_col(times, min(self.window_s, lbl.end_time - t0))
            f_lo = lbl.start_frequency if lbl.start_frequency > 0 else self.fcut_min
            f_hi = lbl.end_frequency if lbl.end_frequency > 0 else self.fcut_max
            y0 = freq_to_row(freqs, min(f_lo, f_hi))
            y1 = freq_to_row(freqs, max(f_lo, f_hi))
            if x1 <= x0:
                x1 = min(x0 + 1, image.shape[1] - 1)
            if y1 <= y0:
                y1 = min(y0 + 1, image.shape[0] - 1)
            if x1 > x0 and y1 > y0:
                boxes.append([float(x0), float(y0), float(x1), float(y1)])

        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
