import torch
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    # Whether the model can serve as a CQR quantile regressor — i.e. its final
    # layer is sized by ``output_dim`` so the runner can widen it to 3*F to emit
    # (q_lo, q_mid, q_hi) per feature.  Set False for models whose output width
    # is structurally locked to input_dim (autoregressive models that feed the
    # prediction back into the input) or that emit their own distribution.
    cqr_compatible = True

    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])


class BaseODModel(BaseModel):
    """Base class for origin-destination (OD) matrix forecasting models.

    **Data contract.** The OD datasets are ``(T, N, N, D)`` — for every
    origin-destination pair there are ``D`` *mobility channels* (e.g. taxi /
    fhv / bike).  The dataloader therefore yields 5-D batches

        X      : (B, T, N, N, D)
        label  : (B, horizon, N, N, D)

    Published OD models (STGCN, GWNET, AGCRN, ODMixer, STZINB, GMEL, …) are all
    *single-channel*: they treat the ``N`` destinations as the feature axis
    (``input_dim = output_dim = node_num``) and predict one ``N×N`` matrix.
    To run those single-channel backbones on multi-mobility data without
    touching their internals, this base class adopts the **channel-as-batch**
    convention used by multimodal baselines (e.g. Deep-NYC-Taxi-Bike, ITSC'22:
    one shared backbone, modes as channels):

        1. fold the ``D`` channels into the batch dimension  →  (B·D, T, N, N)
        2. run the model's own :meth:`forward_single` on that 4-D tensor, which
           is exactly the legacy single-channel OD format every backbone was
           written for, producing (B·D, horizon, N, N)
        3. unfold the channels back out  →  (B, horizon, N, N, D)

    Channels share weights and are predicted independently (no cross-mobility
    coupling), which keeps every model faithful to its source paper, uniform,
    and low-risk.  Subclasses implement :meth:`forward_single` instead of
    :meth:`forward` and build their layers exactly as in the single-channel
    case (final projection sized by ``output_dim = node_num``).

    A subclass that genuinely wants joint cross-channel modelling can override
    :meth:`forward` directly and ignore the fold/unfold helper.
    """

    # OD models emit a plain (B, H, N, N, D) tensor and are not the CQR
    # quantile regressor by default (the conformal path is feature-wise and
    # designed for flow models); keep them off the --cqr gate unless a
    # subclass opts in.
    cqr_compatible = False

    def forward(self, x, label=None):
        """Channel-as-batch wrapper around :meth:`forward_single`.

        ``x`` is ``(B, T, N, N, D)``; returns ``(B, horizon, N, N, D)``.
        Accepts a 4-D ``(B, T, N, N)`` tensor too (single channel, D=1) so the
        efficiency profiler and any single-channel caller still work.
        """
        x, b, d, squeeze_back = self._fold_channels(x)
        out = self.forward_single(x, label=label)  # (B*D, H, N, N)
        return self._unfold_channels(out, b, d, squeeze_back)

    # -- channel-as-batch helpers (reusable by tuple-returning subclasses) ---

    def _fold_channels(self, x):
        """Fold the D mobility channels into the batch dim.

        ``(B, T, N, N, D)`` -> ``(B*D, T, N, N)``.  Returns
        ``(x_folded, B, D, squeeze_back)``; ``squeeze_back`` is True when the
        input was 4-D (single channel) and the output should drop its channel
        axis again.
        """
        squeeze_back = False
        if x.dim() == 4:  # (B, T, N, N) — treat as a single channel
            x = x.unsqueeze(-1)
            squeeze_back = True
        b, t, n, m, d = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(b * d, t, n, m)
        return x, b, d, squeeze_back

    def _unfold_channels(self, out, b, d, squeeze_back=False):
        """Inverse of :meth:`_fold_channels` for a model output.

        ``(B*D, H, N, N)`` -> ``(B, H, N, N, D)`` (or ``(B, H, N, N)`` when
        ``squeeze_back``)."""
        _, h, n, m = out.shape
        out = out.reshape(b, d, h, n, m).permute(0, 2, 3, 4, 1)
        if squeeze_back:
            out = out.squeeze(-1)
        return out

    @abstractmethod
    def forward_single(self, x, label=None):
        """Run the single-channel backbone.

        ``x`` is ``(B', T, N, N)`` where ``B' = B·D`` (channels folded into the
        batch).  Must return ``(B', horizon, N, N)`` — i.e. the legacy
        single-channel OD output.  Build layers as in the single-channel model
        (``input_dim = output_dim = node_num``).

        Distribution models (STZINB, STTN) that emit a *tuple* of parameter
        tensors instead override :meth:`forward` directly and reuse
        :meth:`_fold_channels` / :meth:`_unfold_channels`.
        """
        raise NotImplementedError
