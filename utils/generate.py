import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../../../"))
sys.path.append(os.path.abspath(__file__ + "/../../"))

from utils.args import get_data_path


def _resolve_device(data=None, device: Optional[Union[str, torch.device]] = None):
    if device is not None:
        return torch.device(device)
    if isinstance(data, torch.Tensor):
        return data.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Scaler & metadata ───────────────────────────────────────────────────


def _scaler_to_meta(scaler) -> dict:
    """Extract serialisable metadata from a fitted scaler."""
    cls_name = type(scaler).__name__
    params = {
        "data_min": scaler.data_min_.tolist() if hasattr(scaler.data_min_, 'tolist') else float(scaler.data_min_),
        "data_max": scaler.data_max_.tolist() if hasattr(scaler.data_max_, 'tolist') else float(scaler.data_max_),
        "use_log1p": bool(getattr(scaler, "use_log1p", False)),
    }
    return {"scaler": cls_name, "scaler_params": params}


def reconstruct_scaler(meta: dict):
    """Rebuild a scaler from ``meta.json`` metadata."""
    params = meta.get("scaler_params", {})
    return MinMaxScaler(
        data_min=params["data_min"],
        data_max=params["data_max"],
        use_log1p=params.get("use_log1p", False),
    )


def _save_meta(base_dir: Path, scaler, data_shape, split_sizes: dict):
    """Save ``meta.json`` alongside ``his.npz``."""
    meta = _scaler_to_meta(scaler)
    meta["data_shape"] = list(data_shape)
    meta["splits"] = split_sizes
    with open(base_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta.json to {base_dir / 'meta.json'}")


class MinMaxScaler:
    """Min-max normalization to [0, 1] with optional per-channel and log1p modes.

    Modes:
        * Global  — ``data_min_``/``data_max_`` are 0-d tensors (legacy behavior).
        * Per-channel — ``data_min_``/``data_max_`` are 1-d tensors of shape ``(D,)``;
          broadcasting expects data with a trailing D axis (e.g. ``(T, N, D)``).
        * log1p — applies ``log1p`` before fit/transform; the stored min/max are
          in log-space, and ``inverse_transform`` undoes both steps.
    """

    def __init__(self, data_min=None, data_max=None, use_log1p=False):
        self.use_log1p = bool(use_log1p)
        if data_min is not None:
            self.data_min_ = torch.as_tensor(data_min, dtype=torch.float32)
            self.data_max_ = torch.as_tensor(data_max, dtype=torch.float32)
        else:
            self.data_min_ = None
            self.data_max_ = None

    def _maybe_log1p(self, data):
        if not self.use_log1p:
            return data
        if isinstance(data, torch.Tensor):
            return torch.log1p(data)
        return np.log1p(data)

    def _maybe_expm1(self, data):
        if not self.use_log1p:
            return data
        if isinstance(data, torch.Tensor):
            return torch.expm1(data)
        return np.expm1(data)

    def fit(self, data, per_channel=False):
        work = self._maybe_log1p(data)
        if per_channel:
            # Reduce over every axis except the trailing D axis -> shape (D,)
            d = work.shape[-1]
            flat = work.reshape(-1, d) if isinstance(work, np.ndarray) else work.reshape(-1, d)
            self.data_min_ = torch.as_tensor(flat.min(axis=0), dtype=torch.float32)
            self.data_max_ = torch.as_tensor(flat.max(axis=0), dtype=torch.float32)
        else:
            self.data_min_ = torch.as_tensor(work.min(), dtype=torch.float32)
            self.data_max_ = torch.as_tensor(work.max(), dtype=torch.float32)
        print(f"MinMaxScaler(log1p={self.use_log1p}) min: {self.data_min_.tolist()}, "
              f"max: {self.data_max_.tolist()}")
        return self

    def _params_for(self, data):
        """Return (dmin, dmax, span) as numpy or torch matching ``data``."""
        if isinstance(data, torch.Tensor):
            dmin = self.data_min_.to(data.device)
            dmax = self.data_max_.to(data.device)
        else:
            dmin = self.data_min_.cpu().numpy()
            dmax = self.data_max_.cpu().numpy()
        span = dmax - dmin
        return dmin, dmax, span

    def transform(self, data):
        data = self._maybe_log1p(data)
        dmin, _, span = self._params_for(data)
        if isinstance(span, torch.Tensor):
            safe = torch.where(span == 0, torch.ones_like(span), span)
            out = (data - dmin) / safe
            return torch.where((span == 0).expand_as(out), torch.zeros_like(out), out)
        safe = np.where(span == 0, 1.0, span)
        out = (data - dmin) / safe
        return np.where(np.broadcast_to(span == 0, out.shape), 0.0, out)

    def inverse_transform(self, data, device=None):
        dev = _resolve_device(data, device)
        if isinstance(data, torch.Tensor):
            dmin = self.data_min_.to(dev)
            span = (self.data_max_ - self.data_min_).to(dev)
        else:
            dmin = self.data_min_.cpu().numpy()
            span = (self.data_max_ - self.data_min_).cpu().numpy()
        out = data * span + dmin
        return self._maybe_expm1(out)


# ── Data splitting & saving ──────────────────────────────────────────────


def _split_by_ratio(data, args, val=0.1, test=0.1):
    """Compute train/val/test index splits for a time series."""
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    min_t = abs(min(x_offsets))
    max_t = abs(data.shape[0] - abs(max(y_offsets)))
    idx = np.arange(min_t, max_t, 1)

    N = len(idx)
    n_val = int(round(val * N))
    n_test = int(round(test * N))
    n_train = N - n_val - n_test

    print(f"Index range: [{min_t}, {max_t}), total={N}, train={n_train}, val={n_val}, test={N - n_train - n_val}")

    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]
    idx_all = idx[:]
    return idx_train, idx_val, idx_test, idx_all


def _save_dataset(base_dir, data, scaler, idx_train, idx_val, idx_test, idx_all):
    """Save processed data, indices, and metadata to *base_dir*."""
    _ensure_dir(base_dir)
    np.savez_compressed(base_dir / "his.npz", data=data)
    _save_indices(base_dir, idx_train, idx_val, idx_test, idx_all)
    _save_meta(base_dir, scaler, data.shape, {
        "train": len(idx_train), "val": len(idx_val), "test": len(idx_test),
    })
    print(f"Saved to {base_dir}")


def _save_indices(base_dir: Path, idx_train, idx_val, idx_test, idx_all):
    np.save(base_dir / "idx_train", idx_train)
    np.save(base_dir / "idx_val", idx_val)
    np.save(base_dir / "idx_test", idx_test)
    np.save(base_dir / "idx_all", idx_all)


def _compute_stats(data, label="data"):
    """Compute summary statistics for a numpy array."""
    stats = {
        "label": label,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "median": float(np.median(data)),
        "nonzero_ratio": float(np.count_nonzero(data) / data.size),
        "size": int(data.size),
    }
    return stats


def _save_info(base_dir: Path, raw_stats, transformed_stats, scaler_meta,
               split_sizes, args_dict):
    """Save ``info.json`` with raw/transformed data statistics and config."""
    info = {
        "raw_data": raw_stats,
        "transformed_data": transformed_stats,
        "scaler": scaler_meta,
        "splits": split_sizes,
        "config": args_dict,
    }
    path = base_dir / "info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved info.json to {path}")


# ── Dimension reordering ─────────────────────────────────────────────────


def _reorder_to_time_first(data, fmt="NDT"):
    """Reorder *data* so that T (time) is the first axis.

    *fmt* is a string of dimension labels (case-insensitive):
        N = spatial (region), T = time, D = feature dimension.
    Two consecutive N's indicate an OD matrix (e.g. ``"NNDT"``).

    If the input has fewer dimensions than *fmt*, a D=1 axis is appended.
    Default format is ``"NDT"`` → output ``(T, N, D)``.
    """
    fmt = fmt.upper()

    # If format has no D, add a trailing D=1 axis
    if "D" not in fmt:
        data = data[..., np.newaxis]
        fmt = fmt + "D"

    # If input still has fewer dims than format, pad with trailing size-1 axes
    while data.ndim < len(fmt):
        data = data[..., np.newaxis]

    if len(fmt) != data.ndim:
        raise ValueError(
            f"Format '{fmt}' has {len(fmt)} dims but data has {data.ndim} dims"
        )

    t_pos = fmt.index("T")
    perm = [t_pos] + [i for i in range(len(fmt)) if i != t_pos]
    data = data.transpose(perm)

    new_fmt = "T" + fmt[:t_pos] + fmt[t_pos + 1:]
    print(f"Reordered {fmt} → {new_fmt}, shape: {data.shape}")
    return data


# ── Unified generate ─────────────────────────────────────────────────────


def generate(args):
    """Unified data generation for flow and OD data.

    Loads a ``.npy`` file, reorders dimensions so T is first, applies
    MinMaxScaler, and saves ``his.npz`` + ``meta.json`` + ``info.json`` + index files.
    """
    data = np.load(args.data_path)
    print(f"Loaded {args.data_path}, raw shape: {data.shape}")

    if args.clip_neg:
        data[data < 0] = 0
        print("Clipped negative values to 0")

    data = _reorder_to_time_first(data, args.fmt)
    raw_stats = _compute_stats(data, label="raw (after reorder)")
    print(f"Raw — max: {data.max()}, min: {data.min()}, mean: {data.mean():.4f}, std: {data.std():.4f}")

    idx_train, idx_val, idx_test, idx_all = _split_by_ratio(data, args)

    scaler = MinMaxScaler(use_log1p=getattr(args, "log1p", False))
    scaler.fit(data, per_channel=getattr(args, "per_channel", False))
    data = scaler.transform(data)
    label = f"transformed (MinMaxScaler, per_channel={getattr(args, 'per_channel', False)}, log1p={scaler.use_log1p})"
    transformed_stats = _compute_stats(data, label=label)
    print(f"Normalized — max: {data.max():.4f}, min: {data.min():.4f}, mean: {data.mean():.4f}, std: {data.std():.4f}")

    base_dir = Path(get_data_path()) / args.dataset / args.years
    _save_dataset(base_dir, data, scaler, idx_train, idx_val, idx_test, idx_all)

    split_sizes = {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)}
    args_dict = {
        "data_path": args.data_path,
        "fmt": args.fmt,
        "clip_neg": args.clip_neg,
        "dataset": args.dataset,
        "years": args.years,
        "seq_length_x": args.seq_length_x,
        "seq_length_y": args.seq_length_y,
        "per_channel": getattr(args, "per_channel", False),
        "log1p": scaler.use_log1p,
    }
    _save_info(base_dir, raw_stats, transformed_stats,
               _scaler_to_meta(scaler), split_sizes, args_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False, help="path to raw .npy file")
    parser.add_argument("--fmt", type=str, default="NDT",
                        help="dimension format, e.g. NDT, NTD, NNDT, NT (default: NDT)")
    parser.add_argument("--clip_neg", action="store_true", help="clip negative values to 0")
    parser.add_argument("--dataset", type=str, default="nyc_mobility", help="dataset name")
    parser.add_argument("--years", type=str, default="2025")
    parser.add_argument("--seq_length_x", type=int, default=12, help="input sequence length")
    parser.add_argument("--seq_length_y", type=int, default=1, help="prediction horizon")
    parser.add_argument("--per_channel", action="store_true",
                        help="fit one (min, max) per trailing D channel (recommended when D>1 and channels have very different magnitudes)")
    parser.add_argument("--log1p", action="store_true",
                        help="apply log1p before scaling; inverse_transform undoes both steps")

    args = parser.parse_args()
    generate(args)

