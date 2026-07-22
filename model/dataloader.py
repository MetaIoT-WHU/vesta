import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def downsample_signal(signal: np.ndarray, target_length: int = 100) -> np.ndarray:
    """Resample a signal to a fixed length."""
    current_length = signal.shape[0]
    if current_length > target_length:
        idx = np.linspace(0, current_length - 1, target_length).astype(int)
        return signal[idx, :]
    if current_length < target_length:
        x_old = np.linspace(0, current_length - 1, current_length)
        x_new = np.linspace(0, current_length - 1, target_length)
        interpolated = np.zeros((target_length, signal.shape[1]), dtype=signal.dtype)
        for i in range(signal.shape[1]):
            interpolated[:, i] = np.interp(x_new, x_old, signal[:, i])
        return interpolated
    return signal


def normalize_signal_channels(signal_data: np.ndarray, normalize_amplitude=False, normalize_phase=False, eps: float = 1e-6):
    if signal_data.size == 0:
        return signal_data.astype(np.float32)
    normalized = signal_data.astype(np.float32, copy=True)
    for channel_idx, should_normalize in enumerate([normalize_amplitude, normalize_phase]):
        if should_normalize and channel_idx < normalized.shape[-1]:
            channel = normalized[..., channel_idx]
            mean = np.mean(channel, axis=-1, keepdims=True)
            std = np.std(channel, axis=-1, keepdims=True)
            normalized[..., channel_idx] = (channel - mean) / np.maximum(std, eps)
    return normalized.astype(np.float32)


class ExportedSatelliteSignalDataset(Dataset):
    """Load samples from an exported manifest and npz files."""

    def __init__(
        self,
        split_dir: str,
        max_length: int = 100,
        signal_channels: int = 2,
        normalize_amplitude: bool = True,
        normalize_phase: bool = True,
        normalization_eps: float = 1e-6,
    ):
        self.split_dir = Path(split_dir)
        self.max_length = max_length
        self.signal_channels = signal_channels
        self.normalize_amplitude = normalize_amplitude
        self.normalize_phase = normalize_phase
        self.normalization_eps = normalization_eps
        with open(self.split_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.samples = manifest.get("files", [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        metadata_path = self.split_dir / sample_info["metadata_file"]
        array_path = self.split_dir / sample_info["array_file"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        arrays = np.load(array_path)
        satellite_signals = arrays["satellite_signals_model_input"].astype(np.float32)
        num_satellites = int(metadata.get("num_satellites", satellite_signals.shape[0]))
        satellite_signals = normalize_signal_channels(
            satellite_signals,
            normalize_amplitude=self.normalize_amplitude,
            normalize_phase=self.normalize_phase,
            eps=self.normalization_eps,
        )
        gt_label = metadata.get("gt_label", [0, 0, 0])
        return {
            "satellite_signals": torch.FloatTensor(satellite_signals[:num_satellites]),  # amp + phase per satellite
            "elevations": torch.FloatTensor(np.asarray(metadata.get("elevations", []), dtype=np.float32)[:num_satellites]),  # satellite elevation
            "azimuths": torch.FloatTensor(np.asarray(metadata.get("azimuths", []), dtype=np.float32)[:num_satellites]),  # satellite azimuth
            "label": torch.LongTensor(gt_label[:3]),  # multi-target activity ids (target1, target2, ...)
            "gt_pos": torch.LongTensor([metadata.get("gt_pos", 0)]),  # seat / position id
            "fs": torch.FloatTensor([metadata.get("fs", 20)]),  # sampling rate in Hz
            "num_satellites": torch.LongTensor([num_satellites]),  # valid satellite count
        }


def make_collate_fn(max_length: int = 100) -> Callable:
    """Build a collate function that resamples sequences to ``max_length``."""

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_satellites = max(item["num_satellites"].item() for item in batch)
        batch_size = len(batch)
        non_empty_signal_shapes = [
            item["satellite_signals"].shape for item in batch if item["satellite_signals"].numel() > 0
        ]
        signal_shape = non_empty_signal_shapes[0] if non_empty_signal_shapes else (1, max_length, 2)
        feature_dims = signal_shape[-1]
        if max_satellites == 0:
            return {
                "satellite_signals": torch.zeros(batch_size, 1, max_length, feature_dims),
                "elevations": torch.zeros(batch_size, 1),
                "azimuths": torch.zeros(batch_size, 1),
                "attention_mask": torch.zeros(batch_size, 1, dtype=torch.bool),
                "labels": torch.stack([item["label"] for item in batch]),
                "gt_pos": torch.stack([item["gt_pos"] for item in batch]),
                "fs": torch.stack([item["fs"] for item in batch]),
                "num_satellites": torch.stack([item["num_satellites"] for item in batch]),
            }

        for item in batch:
            num_sats = item["num_satellites"].item()
            if num_sats > 0:
                processed_signals = []
                for sig in item["satellite_signals"]:
                    sig_np = sig.numpy() if isinstance(sig, torch.Tensor) else sig
                    processed_signals.append(
                        torch.tensor(downsample_signal(sig_np, target_length=max_length), dtype=torch.float32)
                    )
                item["satellite_signals"] = torch.stack(processed_signals)

        padded_signals = torch.zeros(batch_size, max_satellites, max_length, feature_dims)
        padded_elevations = torch.zeros(batch_size, max_satellites)
        padded_azimuths = torch.zeros(batch_size, max_satellites)
        attention_mask = torch.zeros(batch_size, max_satellites, dtype=torch.bool)
        for i, item in enumerate(batch):
            num_sats = item["num_satellites"].item()
            if num_sats > 0:
                padded_signals[i, :num_sats, :, :] = item["satellite_signals"]
                padded_elevations[i, :num_sats] = item["elevations"]
                padded_azimuths[i, :num_sats] = item["azimuths"]
                attention_mask[i, :num_sats] = True
        return {
            "satellite_signals": padded_signals,
            "elevations": padded_elevations,
            "azimuths": padded_azimuths,
            "attention_mask": attention_mask,
            "labels": torch.stack([item["label"] for item in batch]),
            "gt_pos": torch.stack([item["gt_pos"] for item in batch]),
            "fs": torch.stack([item["fs"] for item in batch]),
            "num_satellites": torch.stack([item["num_satellites"] for item in batch]),
        }

    return collate_fn


def create_exported_data_loader(
    split_dir: str,
    batch_size: int = 16,
    max_length: int = 100,
    normalize_amplitude: bool = True,
    normalize_phase: bool = True,
    normalization_eps: float = 1e-6,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    dataset = ExportedSatelliteSignalDataset(
        split_dir=split_dir,
        max_length=max_length,
        signal_channels=2,
        normalize_amplitude=normalize_amplitude,
        normalize_phase=normalize_phase,
        normalization_eps=normalization_eps,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(max_length),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
