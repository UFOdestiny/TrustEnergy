"""Shared experiment runner that eliminates main.py boilerplate.

Each model's ``main.py`` provides only what is unique — model-specific
arguments, model construction, and optional callbacks for setup /
optimizer / scheduler / dataloader.  Everything else (seed, device,
data loading, engine creation, train/eval dispatch) is handled here.

Usage example (minimal)::

    from base.runner import run_experiment
    from my_model import MyModel

    def add_args(parser):
        parser.add_argument("--hidden", type=int, default=64)

    def build_model(args, node_num, **ctx):
        return MyModel(node_num=node_num, hidden=args.hidden, ...)

    if __name__ == "__main__":
        run_experiment(model_name="MyModel", add_args=add_args, build_model=build_model)
"""

import os
import sys

import torch

sys.path.append(os.path.abspath(__file__ + "/../../../"))

from base.engine import BaseEngine, BaseEngine_OD
from base.CQR_engine import CQR_Engine
from base.OD_CQR_engine import OD_CQR_Engine
from base.efficiency import profile_efficiency
from utils.args import (
    get_public_config,
    get_log_path,
    print_args,
    resolve_engine_template,
    set_seed,
)
from utils.dataloader import load_dataset, get_dataset_info
from utils.log import get_logger


def _default_optimizer(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)


_FALLBACK_SEQ_LEN = 12
_FALLBACK_HORIZON = 1


def _auto_fill_from_info(args, data_path, logger):
    """Fill ``args.seq_len`` / ``args.horizon`` / ``args.input_dim`` / ``args.output_dim`` from info.json if not set."""
    import json
    info_path = os.path.join(data_path, args.years, "info.json")
    cfg, shape = {}, None
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = json.load(f)
        cfg = info.get("config", {})
        shape = info.get("raw_data", {}).get("shape")

    if args.seq_len is None:
        args.seq_len = cfg.get("seq_length_x", _FALLBACK_SEQ_LEN)
        logger.info(f"Auto seq_len={args.seq_len} (from {'info.json' if 'seq_length_x' in cfg else 'fallback'})")
    if args.horizon is None:
        args.horizon = cfg.get("seq_length_y", _FALLBACK_HORIZON)
        logger.info(f"Auto horizon={args.horizon} (from {'info.json' if 'seq_length_y' in cfg else 'fallback'})")
    if args.input_dim is None:
        if shape and len(shape) >= 3:
            args.input_dim = shape[-1]
            logger.info(f"Auto input_dim={args.input_dim} (from info.json data shape {shape})")
        else:
            args.input_dim = 1
            logger.info(f"Auto input_dim=1 (fallback)")
    if args.output_dim is None:
        args.output_dim = args.input_dim
        logger.info(f"Auto output_dim={args.output_dim} (= input_dim)")


# Sentinel: pass as make_optimizer to disable optimizer (statistical models)
NO_OPTIMIZER = "NO_OPTIMIZER"


def run_experiment(
    model_name,
    add_args,
    build_model,
    *,
    loss_fn="MAE",
    engine_cls=None,
    engine_quantile_cls=None,
    init_weights=False,
    metric_list=None,
    setup=None,
    make_optimizer=None,
    make_scheduler=None,
    load_data=None,
    engine_extras=None,
    device_override=None,
    train_with_export=False,
    od=False,
    od_cqr=False,
):
    """Run a full training / evaluation experiment.

    Parameters
    ----------
    model_name : str
        Name used for logging directories and checkpoint files.
    add_args : callable(parser) -> None
        Adds model-specific CLI arguments to the shared argument parser.
    build_model : callable(args, node_num, **ctx) -> nn.Module
        Constructs and returns the model.  ``ctx`` contains any extra
        data returned by the ``setup`` callback (e.g. adjacency tensors).
    loss_fn : str
        Loss function identifier understood by :class:`base.metrics.Metrics`.
    engine_cls : type, optional
        Engine class for point-prediction mode (default: ``BaseEngine``).
    engine_quantile_cls : type, optional
        Engine class for quantile mode (default: ``CQR_Engine``).
    init_weights : bool
        If *True*, apply Xavier weight initialisation inside the engine.
    metric_list : list[str], optional
        Metrics to track (default: ``["MAE", "MAPE", "RMSE"]``).
    setup : callable, optional
        ``(args, data_path, adj_path, node_num, device, logger) -> dict``
        Pre-model setup (e.g. load adjacency, compute Chebyshev poly).
        The returned dict is forwarded as ``**ctx`` to ``build_model``.
    make_optimizer : callable, optional
        ``(model, args) -> optimizer``.  Default: ``Adam(lr, weight_decay)``.
        Pass ``None`` explicitly to create an engine with ``optimizer=None``
        (used by non-neural statistical models).
    make_scheduler : callable, optional
        ``(optimizer, args) -> scheduler``.  Default: no scheduler.
    load_data : callable, optional
        ``(data_path, args, logger) -> (dataloader, scaler)``.
        Default: :func:`utils.dataloader.load_dataset`.
    engine_extras : dict or callable, optional
        Extra keyword arguments forwarded to the engine constructor.
        If callable, it is called as ``engine_extras(args)`` and should
        return a dict.
    device_override : str or callable, optional
        Override for device selection.  A string (e.g. ``"cpu"``) or a
        callable ``(args) -> str``.
    train_with_export : bool
        If *True*, call ``engine.train(args.export)`` instead of
        ``engine.train()``.  Used by statistical model engines.
    od : bool
        If *True*, this is an origin-destination model.  The runner then
        (a) forces ``args.input_dim = args.output_dim = node_num`` after
        auto-fill — OD data is ``(T, N, N, D)`` so the auto-filled
        ``input_dim`` (= ``D`` mobility channels) is *not* what the
        single-channel backbone expects (it treats the ``N`` destinations as
        features), and (b) defaults ``engine_cls`` to
        :class:`base.engine.BaseEngine_OD`.  Centralising this removes the
        per-model ``setup()`` overrides that used to duplicate it.
    """
    if engine_cls is None:
        engine_cls = BaseEngine_OD if od else BaseEngine
    if engine_quantile_cls is None:
        engine_quantile_cls = CQR_Engine
    if metric_list is None:
        metric_list = ["MAE", "MAPE", "MSE", "RMSE"]
    if load_data is None:
        load_data = load_dataset

    # --- Parse arguments ---
    parser = get_public_config()
    add_args(parser)
    args = parser.parse_args()
    args.model_name = model_name
    if args.cqr != "no":
        args.model_name += "_CQR"

    log_dir = get_log_path(args)
    logger = get_logger(log_dir, __name__)

    set_seed(args.seed)

    # --- Device ---
    if device_override is not None:
        dev_str = device_override(args) if callable(device_override) else device_override
        device = torch.device(dev_str)
    else:
        device = torch.device(args.device)

    # --- Data ---
    data_path, adj_path, node_num = get_dataset_info(args.dataset, args.years)

    # Auto-fill seq_len / horizon from info.json when not specified via CLI
    _auto_fill_from_info(args, data_path, logger)

    # OD models treat the N×N matrix as N features (destinations) with the D
    # mobility channels folded into the batch (see base.model.BaseODModel), so
    # input_dim/output_dim must be node_num — not the auto-filled D channels.
    if od:
        args.input_dim = node_num
        args.output_dim = node_num
        logger.info(f"OD model: input_dim = output_dim = node_num = {node_num}")
        # OD-CQR is post-hoc split conformal around an existing OD point/ZINB
        # prediction.  It keeps the output width unchanged; all other OD
        # models retain the previous explicit rejection.
        if args.cqr != "no":
            if not od_cqr:
                raise ValueError(
                    f"{model_name} is an OD model and does not support --cqr "
                    f"(its output is an OD matrix, not output_dim-driven quantiles)."
                )
            if args.od_calibration == "aci":
                from base.OD_ACI_engine import OD_ACI_Engine
                engine_quantile_cls = OD_ACI_Engine
            else:
                engine_quantile_cls = OD_CQR_Engine

    dataloader, scaler = load_data(data_path, args, logger)

    # --- Optional pre-model setup ---
    ctx = {}
    if setup is not None:
        ctx = setup(args, data_path, adj_path, node_num, device, logger) or {}

    # --- CQR multi-head output ---
    # In CQR mode the model itself becomes the quantile regressor: its final
    # projection emits 3 channels per feature (q_lo, q_mid, q_hi).  Widen
    # output_dim *before* build_model so the model is sized accordingly; the
    # engine reads args.cqr_channels to recover the true feature count F.
    if args.cqr != "no" and not od:
        args.cqr_channels = args.output_dim
        args.output_dim = args.output_dim * 3
        logger.info(
            f"CQR multi-head: output_dim {args.cqr_channels} -> {args.output_dim} "
            f"(q_lo / q_mid / q_hi per feature)"
        )

    # Print args after auto-fill and setup so all values are final
    print_args(logger, args)

    # --- Select engine class ---
    engine_template = resolve_engine_template(
        args, engine_cls, engine_quantile_cls
    )

    # --- Build model ---
    model = build_model(args, node_num, **ctx)

    # CQR widens output_dim to 3*F and reads (q_lo, q_mid, q_hi) from the
    # model's own output.  Models whose output width is locked to input_dim
    # (autoregressive) or that emit their own distribution (e.g. UQGNN) cannot
    # act as the quantile regressor.
    if args.cqr != "no" and not od and not getattr(model, "cqr_compatible", True):
        raise ValueError(
            f"{model_name} does not support --cqr: its output width is not "
            f"driven by output_dim (it is autoregressive or emits its own "
            f"distribution). Run it without --cqr, or use --cqr on a "
            f"point-prediction model whose final layer is sized by output_dim."
        )

    # --- Optimizer ---
    if make_optimizer is NO_OPTIMIZER:
        optimizer = None
    elif make_optimizer is not None:
        optimizer = make_optimizer(model, args)
    else:
        optimizer = _default_optimizer(model, args)

    # --- Scheduler ---
    scheduler = make_scheduler(optimizer, args) if make_scheduler is not None else None

    # --- Engine ---
    engine_kwargs = dict(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_norm=args.clip_grad_norm,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        normalize=args.normalize,
        metric_list=metric_list,
        init_weights=init_weights,
        args=args,
    )
    if engine_extras:
        if callable(engine_extras):
            engine_extras = engine_extras(args)
        engine_kwargs.update(engine_extras)

    engine = engine_template(**engine_kwargs)

    # --- Train / Evaluate ---
    if args.mode == "train":
        if train_with_export:
            engine.train(args.export)
        else:
            engine.train()
    else:
        engine.evaluate(args.mode, args.model_path, args.export)

    # --- Efficiency profiling (after test) ---
    profile_efficiency(engine, dataloader, device, logger, args)
