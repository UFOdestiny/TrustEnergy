"""TrustEnergy — MASTGNN: Memory-Augmented Spatio-Temporal GNN.

Implementation of the *Hierarchical Spatiotemporal Representation* module
of TrustEnergy (arXiv:2601.13422).  The uncertainty side
of the paper (Sequential Conformalized Quantile Regression) lives in
``src/flow/trustenergy/SCQR_engine.py``; this file is only the predictor / quantile
regressor.

Key ideas from the paper realised here
---------------------------------------
* **Memory-augmented parameter pools** (Methodology §"Parameter Pool
  Construction" / "Parameter Updating Mechanism").  Instead of learning a
  full per-node / per-step weight tensor, we keep small shared *pools*
  ``P_s`` and ``P_t`` and *generate* the diffusion-conv kernels from
  compact spatial / temporal embeddings::

      W_s = E_s · P_s ,   W_t = E_t · P_t        (paper Eq. for meta-params)

  This is the node-adaptive weight generation of AGCRN, extended with a
  temporal pool, and reduces the effective parameter footprint from
  ``O(T·N)`` to ``O(k)`` with ``k = B + N``.

* **Memory-augmented recurrent unit** (paper's r_t / u_t / c_t gated cell).
  A diffusion-graph-convolution GRU whose gate / candidate kernels are the
  generated meta-parameters ``Θ_r, Θ_u, Θ_c``.

* **Hierarchical spatial mixing.**  In this single-node-set framework we
  realise the macro/micro hierarchy as a two-view graph: the provided
  geographic adjacency (macro, "nearby users") *plus* a learned
  behavioural-similarity graph ``softmax(relu(E_s E_sᵀ))`` (micro, "users
  with similar usage"), both consumed by the diffusion convolution.

* **Quantile output head.**  A single projection emits ``output_dim``
  channels per node/horizon, so under ``--cqr`` the runner widens it to
  ``3·F`` and the (SC)QR engine reads ``(q_mid, q_lo, q_hi)`` per feature.
  Hence ``cqr_compatible = True``.

Tensor convention: inputs ``x`` are ``(B, T, N, F)``; the model returns
``(B, horizon, N, output_dim)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import BaseModel


class AdaptiveDiffusionGConv(nn.Module):
    """Diffusion graph convolution with memory-generated kernels.

    The kernel ``W ∈ R^{N×S×C_in×C_out}`` is *generated* from the spatial
    embedding ``E_s`` and the shared pool ``P_s`` (``W = E_s · P_s``), so
    every node gets its own kernel without storing one explicitly.  The
    support set stacks the identity, the learned behavioural-similarity
    graph (micro), and the supplied geographic supports (macro).
    """

    def __init__(self, dim_in, dim_out, n_geo_supports, embed_dim, cheb_k=2):
        super().__init__()
        self.cheb_k = cheb_k
        # total mixing matrices: I + (cheb_k-1) powers of the adaptive graph
        # + the geographic diffusion supports
        self.n_support = cheb_k + n_geo_supports
        # shared spatial pool P_s and its bias pool
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, self.n_support, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        nn.init.xavier_normal_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x, node_embed, adaptive_adj, geo_supports):
        # x: (B, N, C_in); node_embed: (N, d_s); adaptive_adj: (N, N)
        node_num = node_embed.shape[0]

        support_set = [torch.eye(node_num, device=x.device), adaptive_adj]
        for _ in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * adaptive_adj, support_set[-1]) - support_set[-2]
            )
        support_set = support_set[: self.cheb_k]
        support_set.extend(geo_supports)               # macro geographic views
        supports = torch.stack(support_set, dim=0)      # (S, N, N)

        # memory-generated kernels: θ_s = E_s · P_s
        weights = torch.einsum("nd,dsio->nsio", node_embed, self.weights_pool)
        bias = torch.matmul(node_embed, self.bias_pool)  # (N, C_out)

        x_g = torch.einsum("snm,bmc->bsnc", supports, x)  # (B, S, N, C_in)
        x_g = x_g.permute(0, 2, 1, 3)                     # (B, N, S, C_in)
        out = torch.einsum("bnsi,nsio->bno", x_g, weights) + bias
        return out                                        # (B, N, C_out)


class MASTGNNCell(nn.Module):
    """Memory-augmented graph-convolutional GRU cell (paper's r/u/c unit)."""

    def __init__(self, dim_in, hidden_dim, n_geo_supports, embed_dim, cheb_k):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AdaptiveDiffusionGConv(
            dim_in + hidden_dim, 2 * hidden_dim, n_geo_supports, embed_dim, cheb_k
        )
        self.update = AdaptiveDiffusionGConv(
            dim_in + hidden_dim, hidden_dim, n_geo_supports, embed_dim, cheb_k
        )

    def forward(self, x, state, node_embed, adaptive_adj, geo_supports):
        # x: (B, N, C_in); state: (B, N, hidden)
        state = state.to(x.device)
        xs = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(xs, node_embed, adaptive_adj, geo_supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat([x, r * state], dim=-1)
        hc = torch.tanh(self.update(candidate, node_embed, adaptive_adj, geo_supports))
        return z * state + (1 - z) * hc

    def init_hidden(self, batch_size, node_num):
        return torch.zeros(batch_size, node_num, self.hidden_dim)


class MASTGNNEncoder(nn.Module):
    """Stack of memory-augmented GCRN cells scanned over time."""

    def __init__(self, dim_in, hidden_dim, n_geo_supports, embed_dim, cheb_k, num_layers):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        cells = [MASTGNNCell(dim_in, hidden_dim, n_geo_supports, embed_dim, cheb_k)]
        for _ in range(1, num_layers):
            cells.append(
                MASTGNNCell(hidden_dim, hidden_dim, n_geo_supports, embed_dim, cheb_k)
            )
        self.cells = nn.ModuleList(cells)

    def forward(self, x, node_embed, adaptive_adj, geo_supports):
        # x: (B, T, N, C_in)
        B, T = x.shape[0], x.shape[1]
        node_num = x.shape[2]
        cur = x
        for i in range(self.num_layers):
            state = self.cells[i].init_hidden(B, node_num).to(x.device)
            seq = []
            for t in range(T):
                state = self.cells[i](
                    cur[:, t], state, node_embed, adaptive_adj, geo_supports
                )
                seq.append(state)
            cur = torch.stack(seq, dim=1)  # (B, T, N, hidden)
        return cur


class MASTGNN(BaseModel):
    """Memory-Augmented Spatio-Temporal GNN predictor (TrustEnergy backbone).

    Args (model-specific):
        embed_dim:   spatial embedding / pool dimension ``d_s``.
        rnn_unit:    hidden dimension ``h`` of the GCRN.
        num_layers:  number of stacked GCRN layers.
        cheb_k:      diffusion order for the adaptive (micro) graph.
        tcn_kernel:  kernel width of the temporal conv refinement (TCN).
        dropout:     dropout on the pooled representation.
        geo_supports: list of pre-normalised dense geographic supports
                      (macro graph), each ``(N, N)``; may be empty.
    """

    def __init__(
        self,
        embed_dim,
        rnn_unit,
        num_layers,
        cheb_k,
        geo_supports,
        tcn_kernel=3,
        dropout=0.1,
        **args,
    ):
        super().__init__(**args)
        self.rnn_unit = rnn_unit
        self.embed_dim = embed_dim

        # geographic (macro) supports — fixed buffers
        geo_supports = geo_supports or []
        self.n_geo = len(geo_supports)
        for i, s in enumerate(geo_supports):
            self.register_buffer(
                f"geo_{i}", torch.as_tensor(s, dtype=torch.float32), persistent=False
            )

        # spatial memory pool query: per-node embedding E_s  (paper Eq. E_s)
        self.node_embed = nn.Parameter(torch.randn(self.node_num, embed_dim) * 0.05)

        # temporal pool: per-step embedding E_t and pool P_t -> additive
        # temporal meta-feature on the input hidden (paper θ_t = E_t · P_t)
        self.temporal_embed = nn.Parameter(torch.randn(self.seq_len, embed_dim) * 0.05)

        self.input_proj = nn.Linear(self.input_dim, rnn_unit)
        self.temporal_pool = nn.Linear(embed_dim, rnn_unit)

        self.encoder = MASTGNNEncoder(
            rnn_unit, rnn_unit, self.n_geo, embed_dim, cheb_k, num_layers
        )

        # lightweight TCN over time to refine temporal patterns (DCGCN + TCN)
        pad = (tcn_kernel - 1) // 2
        self.tcn = nn.Conv2d(
            rnn_unit, rnn_unit, kernel_size=(tcn_kernel, 1), padding=(pad, 0)
        )
        self.dropout = nn.Dropout(dropout)

        # quantile output head: project last hidden -> horizon * output_dim.
        # Output width is driven by output_dim => CQR/SCQR compatible.
        self.end_conv = nn.Conv2d(
            1, self.horizon * self.output_dim, kernel_size=(1, rnn_unit), bias=True
        )

    def _geo_supports(self):
        return [getattr(self, f"geo_{i}") for i in range(self.n_geo)]

    def forward(self, x, label=None):  # x: (B, T, N, F)
        B, T, N, _ = x.shape

        # input embedding + temporal meta-feature (E_t · P_t), broadcast over N
        h = self.input_proj(x)                                   # (B, T, N, h)
        t_feat = self.temporal_pool(self.temporal_embed[:T])     # (T, h)
        h = h + t_feat.view(1, T, 1, -1)

        # behavioural-similarity (micro) graph from the spatial memory pool
        adaptive_adj = F.softmax(
            F.relu(torch.mm(self.node_embed, self.node_embed.t())), dim=1
        )

        # memory-augmented GCRN over time
        hidden = self.encoder(h, self.node_embed, adaptive_adj, self._geo_supports())

        # TCN refinement: (B, T, N, h) -> (B, h, T, N) -> conv -> back
        z = self.tcn(hidden.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        hidden = F.gelu(z) + hidden
        hidden = self.dropout(hidden)

        # take the last step and project to all horizons
        last = hidden[:, -1:, :, :]                              # (B, 1, N, h)
        pred = self.end_conv(last)                               # (B, H*out, N, 1)
        pred = pred.view(B, self.output_dim, self.horizon, N)
        return pred.permute(0, 2, 3, 1).contiguous()             # (B, H, N, out)
