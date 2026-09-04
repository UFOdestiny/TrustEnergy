#!/usr/bin/env python3
"""Print one model-by-metric result table for each configured dataset.

Edit only the configuration block below, then run::

    python utils/res.py

Expected result layout::

    RESULT_ROOT / project / model / dataset / *.log
"""

from pathlib import Path
import re

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

RESULT_ROOT = Path("/home/dy23a.fsu/st/result")

# Projects are searched from left to right for every model/dataset pair.
PROJECTS = ["Chi_OD", "NYC_OD", "DC_OD", "SF_OD"]

DATASETS = [
    "chicago_od_15min_taxi",
    "chicago_od_15min_tnp",
    "chicago_od_15min_bike",
    "nyc_manhattan_od_15min_taxi",
    "nyc_manhattan_od_15min_fhv",
    "nyc_manhattan_od_15min_bike",
    "dc_od_60min_taxi",
    "dc_od_60min_bike",
    "sf_od_15min_taxi",
    "sf_od_15min_bike",
]

# Names must match model directories under result/<project>/.
MODELS = [
    "PDR",
    "PDR_REG",
    "PDR_REG_POST",
    "PDR_no_context",
    "PDR_no_zone_embed",
    "PDR_no_spatial",
    "PDR_no_moe",
    "STZINB",
    "AGCRN_OD",
    "ASTGCN_OD",
    "GMEL",
    "GWNET_OD",
    "HA_OD",
    "HL_OD",
    "HMDLF",
    "LSTM_OD",
    "ODMixer",
    "STGCN_OD",
    "STGODE_OD",
    "STTN",
]

# Complete result metric set registered by base/metrics.py.
RESULT_METRICS = [
    # "NLL",
    # "MGAU",
    # "Quantile",
    "MSE",
    "MAE",
    "MAPE",
    "RMSE",
    # "KL",
    # "CRPS",
    # "MPIW",
    # "WINK",
    # "COV",
    # "IS",
]

# Numeric fields written by base/efficiency.py. Units are encoded in names;
# FLOPs is normalized to an absolute operation count regardless of log unit.
EFFICIENCY_METRICS = [
    "Params",
    # "Trainable_params",
    "CPU_memory_MB",
    "Inference_ms",
    # "Inference_std_ms",
    # "Full_test_s",
    # "Test_batches",
    "GPU_peak_MB",
    # "GPU_reserved_MB",
    # "GPU_current_MB",
    "FLOPs",
    # "GPU_total_GB",
    # "System_RAM_GB",
]

# Used both as the complete selectable list and as the preferred order for
# automatically discovered result + efficiency columns.
ALL_METRICS = RESULT_METRICS + EFFICIENCY_METRICS

# None: include every metric found in the selected result logs.
# A list: include only those columns, e.g. ["MAE", "MAPE", "RMSE"].
# ALL_METRICS: show the complete framework metric set, including empty columns.
METRICS = ALL_METRICS # None

# How to choose when one model/dataset directory contains multiple logs:
#   "best"   -> lowest BEST_METRIC (default)
#   "latest" -> most recently modified log
LOG_SELECTION = "best"
BEST_METRIC = "MAE"

# Set to a metric name (for example "MAE") to rank table rows, or None to
# preserve MODELS order. Missing models are always placed last.
SORT_BY = None

DECIMALS = 3
OUTPUT_FILE = RESULT_ROOT / "res.txt"  # Set to None to disable file output.


# =============================================================================
# Parsing and table generation
# =============================================================================

AVERAGE_RE = re.compile(r"Average:\s*(.*)")
METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_average(log_path):
    """Return metrics from the last ``Average:`` line in a log."""
    average = None
    with log_path.open(errors="ignore") as log_file:
        for line in log_file:
            match = AVERAGE_RE.search(line)
            if match:
                average = {
                    name: float(value)
                    for name, value in METRIC_RE.findall(match.group(1))
                }
    return average


def parse_efficiency(log_path):
    """Return all numeric fields from the log's Efficiency section."""
    values = {}
    in_efficiency = False

    with log_path.open(errors="ignore") as log_file:
        for line in log_file:
            if "Efficiency" in line and "===" in line:
                in_efficiency = True
                continue
            if in_efficiency and "===" in line:
                break
            if not in_efficiency:
                continue

            if "Total Parameters" in line:
                match = re.search(r"Total Parameters\s*:\s*([\d,]+)", line)
                if match:
                    values["Params"] = float(match.group(1).replace(",", ""))
            elif "Trainable Parameters" in line:
                match = re.search(r"Trainable Parameters\s*:\s*([\d,]+)", line)
                if match:
                    values["Trainable_params"] = float(
                        match.group(1).replace(",", "")
                    )
            elif "CPU Memory (RSS)" in line:
                match = re.search(rf"CPU Memory \(RSS\)\s*:\s*{NUMBER}\s*MB", line)
                if match:
                    values["CPU_memory_MB"] = float(match.group(1))
            elif "Inference (1 batch)" in line and "N/A" not in line:
                match = re.search(rf":\s*{NUMBER}\s*±\s*{NUMBER}\s*ms", line)
                if match:
                    values["Inference_ms"] = float(match.group(1))
                    values["Inference_std_ms"] = float(match.group(2))
            elif "Full Test Inference" in line and "N/A" not in line:
                match = re.search(rf":\s*{NUMBER}\s*s\s*\((\d+)\s*batches\)", line)
                if match:
                    values["Full_test_s"] = float(match.group(1))
                    values["Test_batches"] = float(match.group(2))
            elif "GPU Peak Allocated" in line:
                match = re.search(rf":\s*{NUMBER}\s*MB", line)
                if match:
                    values["GPU_peak_MB"] = float(match.group(1))
            elif "GPU Peak Reserved" in line:
                match = re.search(rf":\s*{NUMBER}\s*MB", line)
                if match:
                    values["GPU_reserved_MB"] = float(match.group(1))
            elif "GPU Current Allocated" in line:
                match = re.search(rf":\s*{NUMBER}\s*MB", line)
                if match:
                    values["GPU_current_MB"] = float(match.group(1))
            elif "FLOPs" in line and "N/A" not in line:
                match = re.search(rf":\s*{NUMBER}\s*([TGMK]?)FLOPs", line)
                if match:
                    scale = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
                    values["FLOPs"] = float(match.group(1)) * scale[match.group(2)]
            elif "GPU Memory" in line:
                match = re.search(rf":\s*{NUMBER}\s*GB", line)
                if match:
                    values["GPU_total_GB"] = float(match.group(1))
            elif "System RAM" in line:
                match = re.search(rf":\s*{NUMBER}\s*GB", line)
                if match:
                    values["System_RAM_GB"] = float(match.group(1))

    return values


def parse_result(log_path):
    """Combine predictive metrics and efficiency metrics from one log."""
    values = parse_average(log_path)
    if values is None:
        return None
    values.update(parse_efficiency(log_path))
    return values


def select_log(folder):
    """Select one log according to LOG_SELECTION."""
    logs = list(folder.glob("*.log"))
    if not logs:
        return None, None

    if LOG_SELECTION == "latest":
        log = max(logs, key=lambda path: path.stat().st_mtime)
        return log, parse_result(log)

    if LOG_SELECTION != "best":
        raise ValueError(
            f"LOG_SELECTION must be 'best' or 'latest', got {LOG_SELECTION!r}"
        )

    candidates = []
    for log in logs:
        values = parse_result(log)
        if values is not None and BEST_METRIC in values:
            candidates.append((values[BEST_METRIC], log, values))

    if candidates:
        _, log, values = min(candidates, key=lambda item: item[0])
        return log, values

    log = max(logs, key=lambda path: path.stat().st_mtime)
    return log, parse_result(log)


def find_result(model, dataset):
    """Find one configured result, respecting PROJECTS search order."""
    for project in PROJECTS:
        folder = RESULT_ROOT / project / model / dataset
        log, values = select_log(folder)
        if log is not None and values is not None:
            return project, log, values
    return None, None, None


def discover_metrics():
    """Return the union of metrics in all selected configured result logs."""
    discovered = []
    for dataset in DATASETS:
        for model in MODELS:
            _, _, values = find_result(model, dataset)
            if values:
                for metric in values:
                    if metric not in discovered:
                        discovered.append(metric)

    known = [metric for metric in ALL_METRICS if metric in discovered]
    custom = [metric for metric in discovered if metric not in ALL_METRICS]
    return known + custom


def dataset_table(dataset, metrics):
    """Build a table whose rows are models and columns are metrics."""
    rows = []
    projects_found = []

    for order, model in enumerate(MODELS):
        project, _, values = find_result(model, dataset)
        row = {"Model": model, "_order": order}
        for metric in metrics:
            row[metric] = values.get(metric) if values else None
        rows.append(row)
        if project and project not in projects_found:
            projects_found.append(project)

    table = pd.DataFrame(rows)
    if SORT_BY:
        if SORT_BY not in metrics:
            raise ValueError(f"SORT_BY={SORT_BY!r} is not included in table metrics")
        table = table.sort_values(
            [SORT_BY, "_order"], na_position="last", kind="stable"
        )
    else:
        table = table.sort_values("_order")

    table = table.drop(columns="_order").set_index("Model")
    return table, projects_found


def format_table(dataset, table, projects):
    project_text = ", ".join(projects) if projects else "no matching project"
    title = f"Dataset: {dataset}  |  Project: {project_text}"
    formatters = {
        metric: (lambda value: f"{value:.0f}")
        for metric in ("Params", "Trainable_params", "Test_batches")
        if metric in table.columns
    }
    if "FLOPs" in table.columns:
        formatters["FLOPs"] = lambda value: f"{value:.3e}"
    body = table.to_string(
        na_rep="-",
        formatters=formatters,
        float_format=lambda value: f"{value:.{DECIMALS}f}",
    )
    return f"{title}\n{'=' * len(title)}\n{body}"


def main():
    if not RESULT_ROOT.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {RESULT_ROOT}")
    if not PROJECTS or not DATASETS or not MODELS:
        raise ValueError("PROJECTS, DATASETS, and MODELS cannot be empty")

    metrics = discover_metrics() if METRICS is None else list(METRICS)
    if not metrics:
        raise ValueError("No metrics were configured or found in result logs")

    rendered = []
    for dataset in DATASETS:
        table, projects = dataset_table(dataset, metrics)
        rendered.append(format_table(dataset, table, projects))

    text = "\n\n".join(rendered) + "\n"
    print(text, end="")

    if OUTPUT_FILE is not None:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(text)
        print(f"\nWritten to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
