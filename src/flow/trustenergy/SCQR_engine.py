"""Sequential Conformalized Quantile Regression (SCQR).

This engine implements the calibration module of TrustEnergy (the
*Sequential Conformalized Quantile Regression* described in
arXiv:2601.13422).  It extends
:class:`base.CQR_engine.CQR_Engine` and reuses its whole training /
quantile-parsing / checkpoint machinery; only the *calibration and
interval construction* are replaced.

Two changes versus vanilla CQR (Romano et al., 2019):

1. **Additive conformal correction** (paper Eqs. 4-6).  The
   nonconformity score is the standard CQR residual (in the original,
   inverse-transformed data space)::

       ε_i = max{ q_lo(x_i) - y_i ,  y_i - q_hi(x_i) }

   and the interval is widened *additively* by the finite-sample
   correction ``Q = Quantile_{(1-α)(1+1/|E|)}(E)``::

       Ĉ(X_t) = [ q_lo - Q ,  q_hi + Q ].

2. **Sequential sliding window** (paper Algorithm "SCQR", Eqs. for the
   rolling set ``E``).  Rather than freezing ``Q`` on the validation
   split, the calibration set ``E`` is a fixed-size window that slides
   over the test stream in temporal order: at each test step the
   realised residual ``ε_t`` is appended and the oldest score dropped,
   so ``Q`` is recomputed from the most recent ``m`` residuals.  This
   tracks distribution shift (seasonal / weather / behavioural) without
   assuming exchangeability.

The validation split seeds the window so the first test steps already
have a meaningful correction; from there the window slides over the test
stream.  ``Q`` is maintained per forecast horizon by default
(``--cqr horizon``) or as a single shared stream (``--cqr global``),
mirroring :class:`CQR_Engine`.

Contrast with :class:`src.flow.energymamba.ACQR_engine.ACQR_Engine` (EnergyMamba): ACQR
uses a *width-normalised* score and a *multiplicative* correction with an
online miscoverage feedback loop; SCQR keeps the plain additive CQR score
and only makes the calibration set sequential.
"""

import torch

from base.CQR_engine import CQR_Engine


class SCQR_Engine(CQR_Engine):
    DEFAULT_METRICS = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "IS", "COV", "F1", "TZR", "KL", "CRPS"]

    def __init__(self, scqr_window: int = 200, **args):
        """
        Args:
            scqr_window: ``m`` — number of most-recent nonconformity scores
                kept in the sliding calibration window used to compute the
                conformal correction ``Q`` at each test step.
        """
        super().__init__(**args)
        # Allow a CLI override via args (set in main.py's add_args).
        self.window = int(getattr(self.args, "scqr_window", scqr_window))

        # Per-horizon (or global) seed scores collected on validation; each is
        # a temporally ordered 1-D tensor.  Filled by calibrate().
        self._seed_scores = None

    # -- nonconformity ------------------------------------------------------

    @staticmethod
    def _scores(lower, upper, label):
        """Standard CQR nonconformity score ``max(q_lo - y, y - q_hi)``."""
        return torch.maximum(lower - label, label - upper)

    @staticmethod
    def _finite_sample_quantile(buf, alpha):
        """Finite-sample CQR correction: the ``(1-α)(1+1/n)`` empirical
        quantile of a 1-D score buffer (Romano et al., Eq. 5), NaN-safe."""
        buf = buf[~torch.isnan(buf)]
        n = buf.numel()
        if n == 0:
            return 0.0
        level = (1.0 - alpha) * (1.0 + 1.0 / n)
        level = float(min(max(level, 0.0), 1.0))
        return float(torch.quantile(buf, level))

    # -- calibration: seed the sliding window from validation --------------

    def calibrate(self, mode="val"):
        """Collect nonconformity scores on the held-out split (in temporal
        order) to seed the sliding window, and store a static fallback ``Q``.

        For ``--cqr horizon`` one ordered stream is kept per horizon; for
        ``--cqr global`` a single stream pools all horizons.
        """
        lower, upper, label = self._collect_quantiles(mode)  # (Nval, T, N, F)
        scores = self._scores(lower, upper, label)

        if self.cqr_mode == "horizon":
            T = scores.shape[1]
            # seed[h]: scores at horizon h, raveled per sample in split order
            self._seed_scores = [scores[:, h].reshape(-1) for h in range(T)]
            q = torch.stack(
                [
                    torch.tensor(self._finite_sample_quantile(s, self.alpha))
                    for s in self._seed_scores
                ]
            )
            self.register_conformal(q)
            q_msg = ", ".join(f"h{h + 1}={v:.4f}" for h, v in enumerate(q.tolist()))
        else:
            self._seed_scores = [scores.reshape(-1)]
            q = torch.tensor(self._finite_sample_quantile(self._seed_scores[0], self.alpha))
            self.register_conformal(q)
            q_msg = f"{float(q):.4f}"

        n = int((~torch.isnan(scores)).sum().item())
        self._logger.info(
            f"SCQR seed on '{mode}' (alpha={self.alpha}, mode={self.cqr_mode}, "
            f"window={self.window}, n={n}): Q0 = {q_msg}"
        )
        return self._conformal_q

    # -- sequential calibration over the test stream -----------------------

    def _init_window(self, h):
        """Initialise a sliding window (most-recent ``m`` seed scores)."""
        if self._seed_scores is not None and len(self._seed_scores) > h:
            seed = self._seed_scores[h]
            seed = seed[~torch.isnan(seed)]
            return seed[-self.window :].tolist()
        return []

    def _sequential_intervals(self, lowers, uppers, labels):
        """Walk the test stream in temporal order, widening each interval by
        the additive correction ``Q_t`` derived from the current window, then
        sliding the window with the realised residuals of this step.

        Shapes: ``lowers/uppers/labels`` are ``(B, T, N, F)`` on CPU.
        Returns calibrated ``(lowers, uppers)`` of the same shape.
        """
        B, T = lowers.shape[0], lowers.shape[1]
        per_horizon = self.cqr_mode == "horizon"

        # One window per horizon (or one shared window pooled across horizons).
        windows = [self._init_window(h if per_horizon else 0) for h in range(T)] \
            if per_horizon else [self._init_window(0)]

        fallback = self._conformal_q
        out_lower = torch.empty_like(lowers)
        out_upper = torch.empty_like(uppers)

        for t in range(B):
            for h in range(T):
                w = windows[h] if per_horizon else windows[0]
                if w:
                    buf = torch.tensor(w)
                    Q_t = self._finite_sample_quantile(buf, self.alpha)
                elif fallback is not None:
                    fq = fallback[h] if (per_horizon and torch.is_tensor(fallback)
                                         and fallback.ndim == 1) else fallback
                    Q_t = float(fq)
                else:
                    Q_t = 0.0

                out_lower[t, h] = lowers[t, h] - Q_t
                out_upper[t, h] = uppers[t, h] + Q_t

                # slide the window with this step's realised residuals
                new = self._scores(lowers[t, h], uppers[t, h], labels[t, h])
                new = new[~torch.isnan(new)].reshape(-1)
                if new.numel():
                    w.extend(new.tolist())
                    if len(w) > self.window:
                        del w[: len(w) - self.window]

        self._logger.info(
            f"SCQR online calibration: steps={B}, horizons={T}, "
            f"final window size={len(windows[0])}"
        )
        return out_lower, out_upper

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        """Same flow as CQR_Engine.evaluate, but the conformal widening is the
        sequential sliding-window procedure instead of a static ``±Q``."""
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            if self._seed_scores is None:
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

        # During training-time validation we only need the pinball/quantile
        # metrics; the sequential walk is reserved for the test stream.
        if mode == "val":
            self._compute_per_horizon(mids, lowers, uppers, labels, "valid")
            return

        lowers, uppers = self._sequential_intervals(lowers, uppers, labels)
        self._compute_per_horizon(mids, lowers, uppers, labels, "test")

        if not train_test:
            with self._logger.no_time():
                self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
            for msg in self.metric.get_test_msg():
                self._logger.info(msg)

        if export:
            self.save_result(mids, lowers, uppers, labels)

    def _compute_per_horizon(self, mids, lowers, uppers, labels, mode_name):
        mask_value = torch.tensor(torch.nan)
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

    # -- checkpoint: persist the validation seed + SCQR hyper-params -------

    def save_model(self, save_path):
        import os

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        q = self._conformal_q
        seed = None
        if self._seed_scores is not None:
            seed = [s.cpu() for s in self._seed_scores]
        state = {
            "model": self.model.state_dict(),
            "conformal_q": q.cpu() if torch.is_tensor(q) else q,
            "cqr_mode": self.cqr_mode,
            "alpha": self.alpha,
            "scqr_seed_scores": seed,
            "scqr_window": self.window,
        }
        torch.save(state, os.path.join(save_path, self._time_model))

    def _load_state_dict(self, checkpoint):
        super()._load_state_dict(checkpoint)
        if isinstance(checkpoint, dict) and checkpoint.get("scqr_seed_scores") is not None:
            self._seed_scores = checkpoint["scqr_seed_scores"]
