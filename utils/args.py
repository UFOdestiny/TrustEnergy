import argparse
import os
import re

import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_public_config():
    """Create argument parser with common training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--calibration_tag", type=str, default="",
        help="Optional method tag used to distinguish exported calibration results.",
    )

    parser.add_argument("--dataset", type=str, default="chicago_15min")
    parser.add_argument("--years", type=str, default="2018")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)

    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=None)

    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", action="store_false", dest="normalize")

    parser.add_argument(
        "--cqr", type=str, default="no", choices=["no", "horizon", "global"],
        help="Conformalized Quantile Regression: 'no' disables it (point model); "
             "'horizon' calibrates one conformal correction Q per forecast step; "
             "'global' uses a single shared Q.",
    )
    parser.add_argument("--quantile_alpha", type=float, default=0.1)
    parser.add_argument(
        "--od_calibration", type=str, default="split", choices=["split", "aci"],
        help="OD post-hoc calibration used with --cqr: fixed split conformal "
             "or adaptive conformal inference (ACI).",
    )
    parser.add_argument(
        "--od_aci_gamma", type=float, default=0.005,
        help="ACI coverage-feedback step size for OD post-hoc calibration.",
    )
    parser.add_argument(
        "--od_aci_calibration_size", type=int, default=200000,
        help="Maximum residuals retained per ACI calibration reference distribution.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument("--not_print_args", action="store_true", default=False)
    parser.add_argument("--proj", type=str, default="")

    return parser


# System path configuration.
#
# Override either root without touching the code:
#     export POPST_DATA=/path/to/datasets
#     export POPST_RESULT=/path/to/result
_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
_DEFAULT_RESULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result")

_SYSTEM_CONFIG = {
    "log_base": os.environ.get("POPST_RESULT", _DEFAULT_RESULT),
    "data_base": os.environ.get("POPST_DATA", _DEFAULT_DATA),
}


def get_log_path(args):
    base_path = _SYSTEM_CONFIG["log_base"]
    return f"{base_path}/{args.proj}/{args.model_name}/{args.dataset}/"


def get_data_path():
    return _SYSTEM_CONFIG["data_base"] + "/"


def _fmt_value(v):
    """Format a value for display, showing shapes for tensors/arrays."""
    if isinstance(v, torch.Tensor):
        return f"Tensor{list(v.shape)}"
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)}"
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (torch.Tensor, np.ndarray)):
        shapes = [f"{type(x).__name__}{list(x.shape)}" for x in v]
        return f"[{', '.join(shapes)}]"
    return v


def print_args(logger, args):
    if args.not_print_args:
        return

    all_vars = vars(args)

    # Fixed groups.  Training also collects common scheduler / regularization
    # knobs (step_size, gamma, dropout) that models declare in their own
    # add_args, so they sit with the other training hyperparameters instead of
    # in a catch-all.
    data_keys = ["dataset", "years", "seq_len", "horizon", "input_dim", "output_dim", "normalize"]
    training_keys = ["bs", "max_epochs", "patience", "lrate", "wdecay", "clip_grad_norm",
                     "dropout", "step_size", "gamma", "seed"]
    cqr_keys = ["cqr", "quantile_alpha", "od_calibration", "od_aci_gamma", "od_aci_calibration_size"]
    system_keys = ["device", "mode", "model_path", "export", "proj", "comment"]

    # Everything else a model declares is a model hyperparameter: list it under
    # "Model" right after model_name, in declaration order.
    fixed = set(data_keys + training_keys + cqr_keys + system_keys)
    fixed.update({"model_name", "not_print_args"})
    model_keys = ["model_name"] + [k for k in all_vars if k not in fixed]

    groups = {
        "Data": data_keys,
        "Model": model_keys,
        "Training": training_keys,
        "CQR": cqr_keys,
        "System": system_keys,
    }

    for group_name, keys in groups.items():
        present = [(k, all_vars[k]) for k in keys if k in all_vars]
        if not present:
            continue
        logger.info(f"--- {group_name} ---")
        for k, v in present:
            logger.info(f"  {k:20s}: {_fmt_value(v)}")


def resolve_engine_template(args, standard_engine, quantile_engine):
    if args.cqr != "no":
        return quantile_engine
    return standard_engine


def set_seed(seed):
    """Set random seed for reproducibility across numpy, torch CPU and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def tuple_type(strings):
    """Convert string to integer tuple. Supports formats: '1,2,3' or '(1,2,3)'."""
    cleaned = re.sub(r"[()\\s]", "", strings)
    try:
        return tuple(map(int, cleaned.split(",")))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {strings}") from e
