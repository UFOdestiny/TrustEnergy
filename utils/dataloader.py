import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import yaml

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

from utils.args import get_data_path
from utils.generate import reconstruct_scaler


# ── PyTorch Dataset + Adapter ────────────────────────────────────────────


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding-window time series.

    Each sample is an ``(x, y)`` pair created from index offsets:
    - ``x = data[idx + x_offsets]``  (input window)
    - ``y = data[idx + y_offsets]``  (prediction horizon)
    """

    def __init__(self, data, indices, seq_len, horizon):
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices)
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i + self.x_offsets]
        y = self.data[i + self.y_offsets]
        return torch.from_numpy(x), torch.from_numpy(y)


def _compute_num_batches(size, batch_size, droplast):
    if droplast:
        return size // batch_size
    return math.ceil(size / batch_size) if batch_size else 0


class LoaderAdapter:
    """Wraps a PyTorch DataLoader to expose the ``.shuffle()`` / ``.get_iterator()``
    interface expected by :class:`base.engine.BaseEngine`.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        logger=None,
        name=None,
    ):
        self.dataset = dataset
        self.bs = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self.size = len(dataset)
        self.num_batch = _compute_num_batches(self.size, batch_size, drop_last)
        if logger:
            loader_name = name or "loader"
            logger.info(
                f"{loader_name:5s} num: {self.size},\tBatch num: {self.num_batch}"
            )

    def shuffle(self):
        pass

    def get_iterator(self):
        loader = TorchDataLoader(
            self.dataset,
            batch_size=self.bs,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        return iter(loader)


# ── Data loading ─────────────────────────────────────────────────────────


def _load_data_dir(data_path, years):
    return Path(data_path) / years


def _load_indices(data_dir, split):
    return np.load(data_dir / f"idx_{split}.npy")


def _load_scaler(data_dir, ptr):
    """Reconstruct scaler from meta.json."""
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {data_dir}; regenerate the dataset with utils/generate.py")
    with open(meta_path) as f:
        meta = json.load(f)
    return reconstruct_scaler(meta)


def load_dataset_plain(data_path, args, logger, drop=False):
    """Load dataset as plain numpy arrays (for statistical models)."""
    data_dir = _load_data_dir(data_path, args.years)
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")
    X = ptr["data"]
    xy = []
    for cat in ["train", "val", "test"]:
        idx = _load_indices(data_dir, cat)
        xy.append(X[idx])
    scaler = _load_scaler(data_dir, ptr)
    return xy, scaler


def load_dataset(data_path, args, logger, drop=False):
    """Load dataset as PyTorch DataLoader dict (for neural models)."""
    data_dir = _load_data_dir(data_path, args.years)
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")

    X = ptr["data"]

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = _load_indices(data_dir, cat)
        ds = TimeSeriesDataset(X, idx, args.seq_len, args.horizon)
        dataloader[f"{cat}_loader"] = LoaderAdapter(
            ds,
            batch_size=args.bs,
            shuffle=(cat == "train"),
            drop_last=drop,
            logger=logger,
            name=cat,
        )

    scaler = _load_scaler(data_dir, ptr)
    return dataloader, scaler


# ── Adjacency helpers ────────────────────────────────────────────────────


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


# ── Dataset registry ─────────────────────────────────────────────────────


_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


@lru_cache(maxsize=1)
def _load_registry():
    with open(_REGISTRY_PATH, "r") as f:
        return yaml.safe_load(f)


def get_dataset_info(dataset, years=None):
    """Return ``(data_path, adj_path, node_num)`` for *dataset*.

    ``data_path`` and ``adj_path`` come from ``utils/registry.yaml``.
    ``node_num`` is read from ``info.json`` (or ``meta.json``) inside the
    data directory; if *years* is not provided, defaults to ``None``.
    """
    base_dir = get_data_path()
    registry = _load_registry()
    if dataset not in registry:
        raise KeyError(
            f"Dataset '{dataset}' not found in registry. "
            f"Available: {', '.join(sorted(registry))}"
        )
    entry = registry[dataset]
    data_path = base_dir + entry["data"]
    adj_path = base_dir + entry["adj"]

    # Read node_num from generated info.json / meta.json
    node_num = None
    if years is not None:
        data_dir = Path(data_path) / years
        info_path = data_dir / "info.json"
        meta_path = data_dir / "meta.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            shape = info.get("raw_data", {}).get("shape") or info.get("transformed_data", {}).get("shape")
            if shape and len(shape) >= 2:
                node_num = shape[1]  # (T, N, ...) → N
        elif meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            shape = meta.get("data_shape")
            if shape and len(shape) >= 2:
                node_num = shape[1]

    return [data_path, adj_path, node_num]
