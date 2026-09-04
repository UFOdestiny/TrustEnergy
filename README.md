# TrustEnergy

**A Unified Framework for Accurate and Reliable User-level Energy Usage Prediction**

[![Venue](https://img.shields.io/badge/AAAI-2026-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/41307)
[![arXiv](https://img.shields.io/badge/arXiv-2601.13422-b31b1b)](https://arxiv.org/abs/2601.13422)
[![DOI](https://img.shields.io/badge/DOI-10.1609%2Faaai.v40i46.41307-orange)](https://doi.org/10.1609/aaai.v40i46.41307)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Most deep learning approaches to energy usage prediction either overlook the spatial correlations across households or fail to scale down to individualized, user-level forecasting. And because energy usage is driven by dynamic and uncertain factors — extreme weather above all — a point forecast alone is not enough to act on. TrustEnergy tackles both with two components:

1. **Hierarchical Spatiotemporal Representation (MASTGNN)** — a memory-augmented spatiotemporal graph neural network that captures macro *and* micro usage patterns without a per-node parameter blowup.
2. **Sequential Conformalized Quantile Regression (SCQR)** — dynamically adjusts uncertainty bounds so prediction intervals stay valid over time, with no assumptions on the underlying data distribution.

Evaluated with an electricity provider in Florida, TrustEnergy achieves a 5.4% gain in prediction accuracy and a 5.7% improvement in uncertainty quantification over state-of-the-art baselines.

## Method

### MASTGNN (`src/flow/trustenergy/trustenergy_model.py`)

| Piece | Detail |
| --- | --- |
| **Memory-augmented parameter pools** | Instead of a full per-node / per-step weight tensor, small shared pools `P_s`, `P_t` *generate* the diffusion-convolution kernels from compact spatial / temporal embeddings (`W_s = E_s · P_s`, `W_t = E_t · P_t`), cutting the effective parameter footprint from `O(T·N)` to `O(k)`, `k = B + N`. |
| **Memory-augmented recurrent unit** (`MASTGNNCell`) | A diffusion-graph-convolutional GRU whose reset / update / candidate kernels `Θ_r, Θ_u, Θ_c` are the generated meta-parameters. |
| **Hierarchical spatial mixing** | Two views feed the same diffusion convolution: the geographic adjacency (macro — "nearby users", supplied as the double-transition supports `D^{-1}A` and `D^{-1}Aᵀ`) plus a learned behavioural-similarity graph `softmax(relu(E_s E_sᵀ))` (micro — "users with similar usage"). |
| **Encoder** | `num_layers` stacked memory-augmented GCRN cells scanned over time. |
| **Output head** | A single projection emitting `output_dim` channels per node and horizon, which makes the model a drop-in quantile regressor (`cqr_compatible = True`). |

`forward(x)` takes `(B, seq_len, N, F)` and returns `(B, horizon, N, output_dim)`.

### SCQR (`src/flow/trustenergy/SCQR_engine.py`)

Under `--cqr`, the runner records the true feature count `F` and widens `output_dim` to `3F`, so the model's own projection emits three ordered channels per feature — `q_mid = c0`, `q_lo = c0 − softplus(c1)`, `q_hi = c0 + softplus(c2)` (no quantile crossing) — trained with the pinball loss. `SCQR_Engine` subclasses the shared `CQR_Engine` and replaces only calibration and interval construction:

1. **Additive conformal correction.** The nonconformity score is the standard CQR residual, computed in the original (inverse-transformed) data space:
   `ε_i = max( q_lo(x_i) − y_i , y_i − q_hi(x_i) )`
   and the interval is widened additively by the finite-sample correction `Q = Quantile_{(1−α)(1+1/|E|)}(E)`:
   `Ĉ(X_t) = [ q_lo − Q , q_hi + Q ]`.
2. **Sequential sliding window.** Rather than freezing `Q` on the validation split, the calibration set `E` is a fixed-size window that slides over the test stream in temporal order: at each step the realised residual is appended and the oldest dropped, so `Q` is recomputed from the most recent `m` residuals. This tracks seasonal, weather and behavioural drift without assuming exchangeability.

The validation split seeds the window so the first test steps already carry a meaningful correction. `Q` is maintained per forecast horizon (`--cqr horizon`) or as a single shared stream (`--cqr global`), and is persisted in the checkpoint.

`--cqr no` (the default) runs the plain point model on the shared `BaseEngine`; `--cqr horizon` / `--cqr global` switches in `SCQR_Engine`. Metrics under `--cqr`: `Quantile`, `MAE`, `MAPE`, `RMSE`, `MPIW`, `IS`, `COV`, `F1`, `TZR`, `KL`, `CRPS`.

> Contrast with [EnergyMamba](https://github.com/UFOdestiny/EnergyMamba)'s AS-CQR: that variant uses a *width-normalized* score with a *multiplicative* correction and an online miscoverage feedback loop; SCQR keeps the plain additive CQR score and only makes the calibration set sequential.

## Repository layout

```
TrustEnergy/
  src/flow/trustenergy/
    trustenergy_model.py  MASTGNN: memory-augmented diffusion-GCRN encoder
    SCQR_engine.py        Sequential CQR calibration engine
    main.py               Entry point: model args, adjacency setup, run_experiment()
  base/                   Shared framework
    runner.py             run_experiment(): the single experiment driver
    model.py              BaseModel contract
    engine.py             Training / validation / test loop, checkpointing, export
    CQR_engine.py         Conformalized Quantile Regression engine (SCQR's parent)
    metrics.py            Point, distributional and interval metrics
    efficiency.py         Hardware info, memory, inference time, FLOPs
  utils/
    args.py               Common CLI arguments, path config, set_seed
    dataloader.py         Dataset / DataLoader, dataset registry lookup
    generate.py           Raw array -> his.npz / info.json / split indices
    registry.yaml         Dataset name -> data & adjacency paths
    graph_algo.py         Adjacency normalizations
    get_adj_mat.py        Build an adjacency matrix from geographic shapefiles
    log.py                Logger
    res.py                Result collection / comparison CLI
  jobs/train.sh           Slurm submission script
```

## Installation

```bash
conda create -n st python=3.10 -y
conda activate st

# PyTorch (CUDA 12.8 build)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Data

Point the framework at your data root (defaults to `<repo>/datasets`):

```bash
export POPST_DATA=/path/to/datasets      # where datasets live
export POPST_RESULT=/path/to/result      # where logs & checkpoints go
```

Each dataset folder has this layout, and every entry is produced by `utils/generate.py`:

```
<POPST_DATA>/<dataset>/
  <adj_name>.npy        Adjacency matrix (N x N)
  <years>/
    his.npz             Normalized data + scaler parameters
    info.json           Shape, scaler, split sizes, seq_length_x / seq_length_y
    meta.json           Scaler parameters and raw data shape
    idx_{train,val,test,all}.npy   Split sample indices
```

Generate it from a raw array of shape `(T, N, F)` and register it in `utils/registry.yaml`:

```bash
python utils/generate.py --data_path /path/to/raw.npy --dataset my_energy --years 2018 --fmt NDT
```

```yaml
my_energy:
  data: my_energy
  adj: my_energy/adj.npy
```

`N` is read from `info.json` at runtime; `seq_len` / `horizon` / `input_dim` / `output_dim` are auto-filled from the same file unless given on the command line.

The user-level energy data used in the paper was provided by a Florida electricity provider and cannot be redistributed.

## Usage

```bash
# Point prediction
python src/flow/trustenergy/main.py --dataset chicago_15min --years 2018

# Uncertainty-aware: SCQR with a per-horizon correction, 90% target coverage
python src/flow/trustenergy/main.py --dataset chicago_15min --years 2018 \
    --cqr horizon --quantile_alpha 0.1

# A single shared correction instead
python src/flow/trustenergy/main.py --dataset chicago_15min --cqr global

# Test from a checkpoint (the calibrated Q is restored with it)
python src/flow/trustenergy/main.py --dataset chicago_15min --cqr horizon \
    --mode test --model_path /path/to/TrustEnergy_CQR_<timestamp>.pt

# Export prediction archives
python src/flow/trustenergy/main.py --dataset chicago_15min --mode test --export
```

Slurm:

```bash
sbatch jobs/train.sh
EXTRA="--cqr horizon" DATASETS="chicago_15min" sbatch jobs/train.sh
```

Compare runs:

```bash
python utils/res.py --path result/MyExperiment
python utils/res.py --log result/MyExperiment/TrustEnergy_CQR/chicago_15min/<timestamp>.log
```

Results land in `result/<proj>/TrustEnergy/<dataset>/<timestamp>.log` (`TrustEnergy_CQR` under `--cqr`) next to the matching `.pt` checkpoint.

## Arguments

### Model (MASTGNN)

| Argument | Default | Description |
| --- | --- | --- |
| `--rnn_unit` | `64` | GCRN hidden size `h` |
| `--num_layers` | `2` | Stacked memory-augmented GCRN layers |
| `--embed_dim` | `10` | Spatial pool embedding dimension `d_s` |
| `--cheb_k` | `2` | Adaptive-graph diffusion order |
| `--tcn_kernel` | `3` | Temporal convolution kernel size |
| `--dropout` | `0.1` | Dropout |

### SCQR (used only with `--cqr`)

| Argument | Default | Description |
| --- | --- | --- |
| `--cqr` | `no` | `no` (point model), `horizon` (one correction per forecast step), `global` (single shared correction) |
| `--quantile_alpha` | `0.1` | Target miscoverage `α`; intervals target `1 − α` coverage |
| `--scqr_window` | `200` | `m` — most-recent residuals kept in the sliding calibration window |

### Training

| Argument | Default | Description |
| --- | --- | --- |
| `--bs` | `64` | Batch size |
| `--max_epochs` | `2000` | Maximum epochs |
| `--patience` | `30` | Early-stopping patience on validation loss |
| `--lrate` | `1e-3` | Learning rate (AdamW) |
| `--wdecay` | `5e-4` | Weight decay |
| `--clip_grad_norm` | `5.0` | Gradient-norm clipping |
| `--step_size` | `200` | StepLR decay interval |
| `--gamma` | `0.95` | StepLR decay factor |
| `--seed` | `2025` | Random seed |

### Data & system

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `chicago_15min` | Dataset name (must exist in `registry.yaml`) |
| `--years` | `2018` | Data sub-folder |
| `--seq_len` / `--horizon` | auto | Input length / forecast steps, auto-filled from `info.json` |
| `--input_dim` / `--output_dim` | auto | Feature counts, auto-filled from `info.json` |
| `--no_normalize` | -- | Disable MinMax normalization (on by default) |
| `--device` | `cuda` | Device |
| `--mode` | `train` | `train` or `test` |
| `--model_path` | -- | Checkpoint to load in test mode |
| `--export` | off | Save prediction archives with the final evaluation |
| `--proj` | -- | Sub-folder name for grouping results |

## Implementation details

Experiments were run on a Linux server (Intel Xeon, 64 GB RAM) with an NVIDIA A100 GPU. The reference results in the paper use PyTorch 2.3.0 / CUDA 11.8; this release is pinned to PyTorch 2.8.0 / CUDA 12.8.

## Baselines

Deterministic baselines follow
[STGCN](https://github.com/hazdzz/STGCN),
[GWNET](https://github.com/nnzhan/Graph-WaveNet),
[ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch),
[AGCRN](https://github.com/LeiBAI/AGCRN),
[StemGNN](https://github.com/microsoft/StemGNN),
[DSTAGNN](https://github.com/SYLan2019/DSTAGNN),
[PDFormer](https://github.com/BUAABIGSCity/PDFormer),
[PowerPM](https://github.com/KimMeen/Time-LLM),
[Chronos](https://github.com/amazon-science/chronos-forecasting), and
[Moment](https://github.com/moment-timeseries-foundation-model/moment).

Probabilistic baselines follow
[STZINB](https://github.com/ZhuangDingyi/STZINB),
[DiffSTG](https://github.com/wenhaomin/DiffSTG), and
[DeepSTUQ](https://github.com/WeizhuQIAN/DeepSTUQ_Pytorch).

Runnable implementations of all of them, on the same runner and metric pipeline used here, are available in [POPST](https://github.com/UFOdestiny/POPST).

## Citation

```bibtex
@inproceedings{yu2026trustenergy,
  title     = {TrustEnergy: A Unified Framework for Accurate and Reliable User-level Energy Usage Prediction},
  author    = {Yu, Dahai and Xu, Rongchao and Zhuang, Dingyi and Bu, Yuheng and Wang, Shenhao and Wang, Guang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {46},
  pages     = {39558--39566},
  year      = {2026},
  doi       = {10.1609/aaai.v40i46.41307}
}
```

*Published in the AAAI 2026 AI for Social Impact Track.*

## Related work

- [POPST](https://github.com/UFOdestiny/POPST) — the unified spatiotemporal benchmarking framework this release is extracted from (~30 flow models, ~17 OD models, shared conformal-prediction engines)
- [EnergyMamba](https://github.com/UFOdestiny/EnergyMamba) (KDD 2026) — graph-enhanced selective state space model with adaptive sequential CQR
- [HealthMamba](https://github.com/UFOdestiny/HealthMamba) (IJCAI 2026) — graph state space model with three-mechanism uncertainty quantification
- [UQGNN](https://github.com/UFOdestiny/UQGNN) (SIGSPATIAL 2025) — multivariate Gaussian spatiotemporal prediction

## License

Released under the [MIT License](LICENSE).
