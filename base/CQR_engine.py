import os

import numpy as np
import torch
import torch.nn.functional as F

from base.engine import BaseEngine
from base.metrics import Metrics


class CQR_Engine(BaseEngine):
    """Conformalized Quantile Regression engine (Romano et al., NeurIPS 2019).

    The model *is* the quantile regressor: in CQR mode the runner widens
    ``output_dim`` to ``3 * F`` (``F`` = true feature count, read back from
    ``args.cqr_channels``), so the model's own final projection emits three
    channels per feature.  Those channels are parameterised into ordered
    quantiles

        q_mid = c0 ,   q_lo = c0 - softplus(c1) ,   q_hi = c0 + softplus(c2)

    which makes ``q_lo <= q_mid <= q_hi`` true by construction (no quantile
    crossing).  Because the whole network — not a bolt-on head — produces the
    quantiles, interval width is a full non-linear function of the input and
    adapts per node / horizon / sample.

    Pipeline:
      1. **Train** the model with the pinball loss for the lower / median /
         upper quantiles (``alpha/2``, ``0.5``, ``1 - alpha/2``).
      2. **Calibrate**: on a held-out split (the validation set, disjoint from
         the gradient-training data) compute the CQR conformity scores
         ``E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))`` and the
         ``ceil((n+1)(1-alpha))/n`` empirical quantile ``Q`` of those scores.
      3. **Predict**: widen to ``[q_lo - Q, q_hi + Q]``, which guarantees
         marginal coverage ``>= 1 - alpha`` in finite samples.

    Quantiles are parameterised in normalised space and then inverse-transformed
    per feature channel; the inverse transform is monotonic, so the ordering is
    preserved and calibration / metrics happen in the original data space.
    ``Q`` is computed per forecast horizon by default (uncertainty typically
    grows with the horizon); pass ``--cqr global`` for a single shared
    correction.
    """

    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV", "F1", "TZR", "KL", "CRPS"]

    def __init__(self, cqr_min_width: float = 0.0, **args):
        """
        Args:
            cqr_min_width: Optional floor (in normalised space) added to the
                lower/upper softplus deltas.  Default 0.0 — conformal
                calibration is responsible for achieving coverage.
        """
        args["loss_fn"] = "Quantile"
        args["metric_list"] = self.DEFAULT_METRICS
        super().__init__(**args)

        # True per-feature count F (model output_dim is 3F in CQR mode).
        self.cqr_channels = int(
            getattr(self.args, "cqr_channels", self.model.output_dim // 3)
        )
        self.min_width = cqr_min_width

        # Target miscoverage and the corresponding quantile levels.
        self.alpha = float(getattr(self.args, "quantile_alpha", 0.1))
        self.q_lower = self.alpha / 2.0
        self.q_upper = 1.0 - self.alpha / 2.0
        # Conformal correction granularity, from --cqr ("horizon" or "global").
        self.cqr_mode = getattr(self.args, "cqr", "horizon")

        # Conformal correction(s); filled by calibrate(), persisted in checkpoints.
        # Shape: scalar tensor ("global") or (horizon,) ("horizon").
        self.register_conformal(None)

        self.metric = Metrics(
            self._loss_fn, args["metric_list"], horizon=self.model.horizon
        )

    # -- conformal state ----------------------------------------------------

    def register_conformal(self, q):
        """Store the conformal correction Q (None until calibrated)."""
        self._conformal_q = q

    def _quantile_kw(self):
        """Quantile levels / miscoverage forwarded to the interval metrics."""
        return {"q_lower": self.q_lower, "q_upper": self.q_upper, "alpha": self.alpha}

    # -- forward ------------------------------------------------------------

    def _parse_quantiles(self, out):
        """Parse a model output of shape (B, T, N, 3F) into ordered quantiles.

        Returns (lower, mid, upper), each (B, T, N, F), in the model's output
        (normalised) space.
        """
        expected = self.cqr_channels * 3
        if out.shape[-1] != expected:
            raise RuntimeError(
                f"CQR expects the model to emit {expected} channels "
                f"(3 x {self.cqr_channels} features) but got {out.shape[-1]}. "
                f"This model's output width is not driven by output_dim — its "
                f"final layer is likely sized by input_dim/feature. Fix the "
                f"model's output layer to use output_dim, or run it without --cqr."
            )
        out = out.reshape(*out.shape[:-1], self.cqr_channels, 3)
        mid = out[..., 0]
        lower = mid - F.softplus(out[..., 1]) - self.min_width
        upper = mid + F.softplus(out[..., 2]) + self.min_width
        return lower, mid, upper

    def _forward_quantiles(self, X, label):
        pred = self._predict(X, label=label, iter=self._iter_cnt)
        if isinstance(pred, tuple):  # models that also return a scale (unused here)
            pred = pred[0]
        lower, mid, upper = self._parse_quantiles(pred)
        if self._normalize:
            # Per-channel inverse transform; monotonic, so ordering is kept.
            lower, mid, upper, label = self._inverse_transform(
                [lower, mid, upper, label], device=self._device.type
            )
        return lower, mid, upper, label

    def _apply_conformal(self, lower, upper):
        """Widen quantile predictions by the calibrated correction Q."""
        if self._conformal_q is None:
            return lower, upper
        q = self._conformal_q
        if torch.is_tensor(q) and q.ndim == 1:
            # per-horizon: broadcast (T,) over (B, T, N, F)
            q = q.to(lower.device).view(1, -1, *([1] * (lower.ndim - 2)))
        else:
            q = float(q)
        return lower - q, upper + q

    # -- training -----------------------------------------------------------

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n"
                    + "=" * 25
                    + "   Training   "
                    + "=" * 25
                )
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])

            lower, mid, upper, label = self._forward_quantiles(X, label)

            res = self.metric.compute_one_batch(
                mid,
                label,
                mask_value,
                "train",
                lower=lower,
                upper=upper,
                **self._quantile_kw(),
            )
            res.backward()

            if self._clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
            self._optimizer.step()
            self._iter_cnt += 1

    # -- conformal calibration ---------------------------------------------

    def _collect_quantiles(self, mode):
        """Run the quantile model over a split; return (lower, upper, label)
        tensors stacked over the whole split, on CPU."""
        self.model.eval()
        lowers, uppers, labels = [], [], []
        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                lower, _, upper, label = self._forward_quantiles(X, label)
                lowers.append(lower.cpu())
                uppers.append(upper.cpu())
                labels.append(label.cpu())
        return (
            torch.cat(lowers, dim=0),
            torch.cat(uppers, dim=0),
            torch.cat(labels, dim=0),
        )

    @staticmethod
    def _conformal_quantile(scores, alpha):
        """The CQR correction: the ceil((n+1)(1-alpha)) / n empirical quantile
        of the conformity scores (NaN-safe)."""
        scores = scores[~torch.isnan(scores)]
        n = scores.numel()
        if n == 0:
            return torch.tensor(0.0)
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = float(min(level, 1.0))  # clip when n is too small for the guarantee
        return torch.quantile(scores, level)

    def calibrate(self, mode="val"):
        """Compute the conformal correction Q on a held-out split.

        Uses the validation split, which is disjoint from the data the model
        is trained on by gradient descent — the exchangeability CQR needs.
        """
        lower, upper, label = self._collect_quantiles(mode)
        # CQR conformity score: E = max(q_lo - y, y - q_hi)
        scores = torch.maximum(lower - label, label - upper)

        if self.cqr_mode == "horizon":
            T = scores.shape[1]
            q = torch.stack(
                [self._conformal_quantile(scores[:, h], self.alpha) for h in range(T)]
            )
            self.register_conformal(q)
            q_msg = ", ".join(f"h{h + 1}={v:.4f}" for h, v in enumerate(q.tolist()))
        else:
            q = self._conformal_quantile(scores, self.alpha)
            self.register_conformal(q)
            q_msg = f"{float(q):.4f}"

        n = (~torch.isnan(scores)).sum().item()
        self._logger.info(
            f"CQR calibration on '{mode}' (alpha={self.alpha}, mode={self.cqr_mode}, "
            f"n={n}): Q = {q_msg}"
        )
        return self._conformal_q

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            # When evaluating a checkpoint directly (not right after train()),
            # calibrate if the loaded model has no stored correction yet.
            if self._conformal_q is None:
                self.calibrate()

        self.model.eval()

        mids, lowers, uppers, labels = [], [], [], []

        with torch.no_grad():
            loader_key = "test_loader" if mode == "export" else f"{mode}_loader"
            for X, label in self._dataloader[loader_key].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                lower, mid, upper, label = self._forward_quantiles(X, label)

                mids.append(mid.cpu())
                lowers.append(lower.cpu())
                uppers.append(upper.cpu())
                labels.append(label.cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        # Apply the conformal correction to obtain calibrated intervals.
        lowers, uppers = self._apply_conformal(lowers, uppers)

        mask_value = torch.tensor(torch.nan)

        def _compute(mode_name):
            for h in range(mids.shape[1]):
                self.metric.compute_one_batch(
                    mids[:, h : h + 1],
                    labels[:, h : h + 1],
                    mask_value,
                    mode_name,
                    lower=lowers[:, h : h + 1],
                    upper=uppers[:, h : h + 1],
                    **self._quantile_kw(),
                )

        if mode == "val":
            _compute("valid")
            return

        if mode in {"test", "export"}:
            _compute("test")

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(mids, lowers, uppers, labels)

    def save_result(self, mids, lowers, uppers, labels):
        result = torch.stack([lowers, mids, uppers, labels], dim=0)
        base_name = f"{self.args.model_name}-{self.args.dataset}-res"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)

        # Append numeric suffix if file already exists
        suffix = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{suffix}.npy"
            path = os.path.join(self._save_path, save_name)
            suffix += 1

        np.save(path, result.numpy())
        self._logger.info(
            f"Results Save Path: {path} | Shape: {result.shape} (component, batch, horizon, node, feature)"
        )

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = self._time_model
        q = self._conformal_q
        state = {
            "model": self.model.state_dict(),
            "conformal_q": q.cpu() if torch.is_tensor(q) else q,
            "cqr_mode": self.cqr_mode,
            "alpha": self.alpha,
        }
        torch.save(state, os.path.join(save_path, filename))

    def _load_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
            if checkpoint.get("conformal_q") is not None:
                self.register_conformal(checkpoint["conformal_q"])
        else:
            self.model.load_state_dict(checkpoint, strict=False)

    def load_model(self, save_path):
        filename = self._time_model
        f = os.path.join(save_path, filename)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i.endswith(".pt")]
            if not models:
                self._logger.info(f"Model {f} Not Exist. No More Models.")
                exit()
            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            f = os.path.join(save_path, models[-1])
            self._logger.info(
                f"Model {filename} Not Exist. Try the Newest Model {os.path.basename(f)}."
            )
        checkpoint = torch.load(f, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)

    def load_exact_model(self, path):
        checkpoint = torch.load(path, weights_only=False, map_location=self._device)
        self._load_state_dict(checkpoint)
