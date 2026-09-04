import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../../"))

import numpy as np
import torch

from base.runner import run_experiment
from src.flow.trustenergy.SCQR_engine import SCQR_Engine
from src.flow.trustenergy.trustenergy_model import MASTGNN
from utils.dataloader import load_adj_from_numpy
from utils.graph_algo import normalize_adj_mx


def add_args(parser):
    # MASTGNN architecture (memory-augmented spatiotemporal GNN)
    parser.add_argument("--embed_dim", type=int, default=10)   # spatial pool dim d_s
    parser.add_argument("--rnn_unit", type=int, default=64)     # GCRN hidden size h
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)        # adaptive-graph diffusion order
    parser.add_argument("--tcn_kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # SCQR calibration (used only under --cqr)
    parser.add_argument("--scqr_window", type=int, default=200)

    # optimisation
    parser.add_argument("--step_size", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)


def setup(args, data_path, adj_path, node_num, device, logger):
    # Macro (geographic) graph: double-transition diffusion supports
    # D^{-1}A and D^{-1}Aᵀ, the standard random-walk supports for the
    # diffusion convolution.  The micro behavioural graph is learned inside
    # the model from the spatial memory pool.
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)                       # drop pre-existing self-loops
    supports = normalize_adj_mx(adj_mx, "doubletransition")  # [(N,N), (N,N)]
    geo = [np.asarray(s, dtype=np.float32) for s in supports]
    return {"geo_supports": geo}


def build_model(args, node_num, **ctx):
    return MASTGNN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layers=args.num_layers,
        cheb_k=args.cheb_k,
        tcn_kernel=args.tcn_kernel,
        dropout=args.dropout,
        geo_supports=ctx["geo_supports"],
    )


if __name__ == "__main__":
    run_experiment(
        model_name="TrustEnergy",
        add_args=add_args,
        build_model=build_model,
        setup=setup,
        engine_quantile_cls=SCQR_Engine,
        device_override="cuda:0",
        make_optimizer=lambda m, a: torch.optim.AdamW(
            m.parameters(), lr=a.lrate, weight_decay=a.wdecay
        ),
        make_scheduler=lambda o, a: torch.optim.lr_scheduler.StepLR(
            o, step_size=a.step_size, gamma=a.gamma
        ),
    )
