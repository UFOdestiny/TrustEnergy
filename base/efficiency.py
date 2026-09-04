"""Efficiency profiling utilities — GPU/CPU memory, inference time, FLOPs.

Called after test evaluation to report resource usage.  All functions are
designed to degrade gracefully: if CUDA is unavailable or a measurement
fails, it simply logs a warning and continues.

Usage in engine / runner::

    from base.efficiency import profile_efficiency
    profile_efficiency(engine, dataloader, device, logger, args)
"""

import os
import platform
import time

import numpy as np
import psutil
import torch


# ---------------------------------------------------------------------------
# Hardware / platform info
# ---------------------------------------------------------------------------

def _platform_info(device, logger):
    """Log hardware and platform information."""
    import sys

    logger.info("--- Platform ---")
    logger.info(f"  OS                 : {platform.system()} {platform.release()}")
    logger.info(f"  Python             : {sys.version.split()[0]}")
    logger.info(f"  PyTorch            : {torch.__version__}")

    # CPU
    try:
        cpu_name = platform.processor() or "Unknown"
        if cpu_name in ("", "Unknown"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":")[1].strip()
                        break
    except Exception:
        cpu_name = platform.processor() or "Unknown"
    cpu_count = psutil.cpu_count(logical=True)
    phys_count = psutil.cpu_count(logical=False)
    logger.info(f"  CPU                : {cpu_name}")
    logger.info(f"  CPU Cores          : {phys_count} physical, {cpu_count} logical")

    # RAM
    mem = psutil.virtual_memory()
    logger.info(f"  System RAM         : {mem.total / 1024**3:.1f} GB")

    # GPU
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        logger.info(f"  GPU                : {gpu_name}")
        logger.info(f"  GPU Memory         : {gpu_mem:.1f} GB")
        logger.info(f"  CUDA Version       : {torch.version.cuda}")
    else:
        logger.info("  GPU                : N/A")


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _gpu_memory_stats(device):
    """Return dict of GPU memory stats in MB, or None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else 0
    return {
        "peak_allocated_MB": torch.cuda.max_memory_allocated(idx) / 1024 ** 2,
        "peak_reserved_MB": torch.cuda.max_memory_reserved(idx) / 1024 ** 2,
        "current_allocated_MB": torch.cuda.memory_allocated(idx) / 1024 ** 2,
    }


def _cpu_memory_mb():
    """Return current process RSS in MB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
# FLOPs estimation (torch.profiler)
# ---------------------------------------------------------------------------

def _run_forward(engine, X, label):
    """Run one forward pass through the engine so each model's own calling
    convention is honoured.

    The profiler must not call ``model(x)`` directly: several models have
    non-standard ``forward`` signatures (DCRNN needs ``(source, target, iter)``,
    DGCRN needs ``(input, label, iter, horizon)``) and would raise on a bare
    ``model(x)``.  Engines encode the correct call in ``_predict``; reuse it so
    inference-time / FLOPs measurements work for every model rather than
    silently coming back empty.
    """
    return engine._predict(X, label=label, iter=0)


def _estimate_flops(engine, sample_X, sample_label, device):
    """Estimate FLOPs for a single forward pass using torch.profiler.

    Returns (flops: int, readable: str) or (None, None) on failure.
    """
    try:
        from torch.profiler import profile, ProfilerActivity

        engine.model.eval()

        with profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                _run_forward(engine, sample_X, sample_label)

        events = prof.key_averages()
        total_flops = sum(e.flops for e in events if e.flops)
        if total_flops == 0:
            return None, None
        return total_flops, _readable_flops(total_flops)
    except Exception:
        return None, None


def _readable_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops:.0f} FLOPs"


# ---------------------------------------------------------------------------
# Inference time
# ---------------------------------------------------------------------------

def _to_device(t, device):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    return t.to(device)


def _measure_inference_time(engine, dataloader, device, n_warmup=3, n_repeat=10):
    """Measure average single-batch inference time in milliseconds.

    Returns (avg_ms, std_ms) or (None, None) on failure.
    """
    engine.model.eval()
    try:
        iterator = dataloader["test_loader"].get_iterator()
        sample_X, sample_label = next(iter(iterator))
        sample_X = _to_device(sample_X, device)
        sample_label = _to_device(sample_label, device)
    except Exception:
        return None, None

    # Warmup
    try:
        with torch.no_grad():
            for _ in range(n_warmup):
                _run_forward(engine, sample_X, sample_label)
    except Exception:
        return None, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _run_forward(engine, sample_X, sample_label)
            except Exception:
                return None, None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def _measure_full_test_time(engine, dataloader, device):
    """Measure total wall-clock time for one full pass over the test set.

    Returns (total_seconds, num_batches) or (None, None).
    """
    engine.model.eval()
    try:
        iterator = dataloader["test_loader"].get_iterator()
    except Exception:
        return None, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    n_batches = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for X, label in iterator:
            X = _to_device(X, device)
            label = _to_device(label, device)
            try:
                _run_forward(engine, X, label)
            except Exception:
                return None, None
            n_batches += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return elapsed, n_batches


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def _count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_efficiency(engine, dataloader, device, logger, args=None):
    """Run all efficiency profiling and log results.

    Call this after test evaluation has completed.  Takes the *engine* (not the
    bare model) so forward passes go through ``engine._predict`` and honour each
    model's calling convention (DCRNN/DGCRN need extra forward args).
    """
    model = engine.model

    with logger.no_time():
        logger.info("")
        logger.info("=" * 25 + "   Efficiency   " + "=" * 25)

    # 0. Platform / hardware info
    _platform_info(device, logger)

    # 1. Parameter count
    total_params, trainable_params = _count_parameters(model)
    logger.info(f"Total Parameters     : {total_params:,}")
    logger.info(f"Trainable Parameters : {trainable_params:,}")

    # 2. Memory
    cpu_mb = _cpu_memory_mb()
    logger.info(f"CPU Memory (RSS)     : {cpu_mb:.1f} MB")

    # Reset the CUDA peak counter so the reported peak reflects *inference*
    # below, not the train+eval that already ran (training holds gradients and
    # optimizer state, which would otherwise dominate the "peak" and mislead).
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        torch.cuda.reset_peak_memory_stats(idx)

    # 3. Inference time (single batch)
    avg_ms, std_ms = _measure_inference_time(engine, dataloader, device)
    if avg_ms is not None:
        logger.info(f"Inference (1 batch)  : {avg_ms:.2f} ± {std_ms:.2f} ms")
    else:
        logger.info("Inference (1 batch)  : N/A")

    # 4. Full test set inference
    total_sec, n_batches = _measure_full_test_time(engine, dataloader, device)
    if total_sec is not None:
        logger.info(f"Full Test Inference  : {total_sec:.3f} s ({n_batches} batches)")
    else:
        logger.info("Full Test Inference  : N/A")

    # 5. GPU memory (peak now reflects the inference passes above)
    gpu_stats = _gpu_memory_stats(device)
    if gpu_stats:
        logger.info(f"GPU Peak Allocated   : {gpu_stats['peak_allocated_MB']:.1f} MB")
        logger.info(f"GPU Peak Reserved    : {gpu_stats['peak_reserved_MB']:.1f} MB")
        logger.info(f"GPU Current Allocated: {gpu_stats['current_allocated_MB']:.1f} MB")

    # 6. FLOPs
    try:
        iterator = dataloader["test_loader"].get_iterator()
        sample_X, sample_label = next(iter(iterator))
        sample_X = _to_device(sample_X, device)
        sample_label = _to_device(sample_label, device)
        flops, flops_str = _estimate_flops(engine, sample_X, sample_label, device)
        if flops is not None:
            logger.info(f"FLOPs (1 forward)    : {flops_str}")
        else:
            logger.info("FLOPs                : N/A (profiler returned 0)")
    except Exception:
        logger.info("FLOPs                : N/A")

    with logger.no_time():
        logger.info("=" * 66)
