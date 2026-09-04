"""Post-hoc split-conformal intervals for point and ZINB OD forecasters.

This is intentionally different from :mod:`base.CQR_engine`: OD models emit
``(B, H, origin, destination, channel)`` matrices and most pre-trained OD
baselines have a *point* head (or a ZINB distribution), not a 3-quantile head.
OD_CQR therefore preserves their centre forecast exactly and calibrates only a
non-negative symmetric residual radius on validation data.  Consequently MAE
and MSE are unchanged by the interval calculation.
"""

import os

import numpy as np
import torch

from base.engine import BaseEngine_OD
from base.metrics import Metrics, zinb_mean


class OD_CQR_Engine(BaseEngine_OD):
    """Generic post-hoc CQR for 5-D OD tensors.

    ``--cqr horizon`` learns one correction per horizon; ``--cqr global``
    pools horizons.  A three-tuple is interpreted as ZINB ``(n,p,pi)``;
    a two-tuple is interpreted as location-scale ``(loc, scale)`` output.
    """

    DEFAULT_METRICS = ["MAE", "MAPE", "MSE", "RMSE", "MPIW", "IS", "COV", "F1", "TZR", "KL", "CRPS"]

    def __init__(self, **args):
        # This engine is evaluation/post-hoc focused.  Do not retain NLL or
        # model-specific losses in the metric tracker: a common point-centred
        # interval metric is required for all six OD architectures.
        args["loss_fn"] = "MAE"
        args["metric_list"] = self.DEFAULT_METRICS
        super().__init__(**args)
        self.alpha = float(getattr(self.args, "quantile_alpha", 0.05))
        if not 0 < self.alpha < 1:
            raise ValueError("quantile_alpha must lie in (0, 1)")
        self.cqr_mode = getattr(self.args, "cqr", "horizon")
        self._conformal_q = None
        self.metric = Metrics("MAE", self.DEFAULT_METRICS, self.model.horizon)

    @staticmethod
    def _finite_quantile(scores, alpha):
        # ``torch.quantile`` has an implementation-size limit on large OD
        # matrices (e.g. 3503 x 77 x 77 validation cells).  Split conformal
        # needs an order statistic, not interpolated quantiles, so numpy's
        # in-place partition is both exact for CQR and scalable here.
        scores = scores[torch.isfinite(scores)].detach().cpu().numpy().reshape(-1)
        n = scores.size
        if n == 0:
            return torch.tensor(0.0)
        rank = min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1
        return torch.tensor(np.partition(scores, rank)[rank])

    def _point_prediction(self, X, label):
        out = self._predict(X, label=label, iter=self._iter_cnt)
        if isinstance(out, tuple):
            if len(out) == 3:
                # ZINB parameters are defined in original count space; only
                # the normalised data label is inverse transformed.
                pred = zinb_mean(*out)
                if self._normalize:
                    label = self._inverse_transform(label, device=self._device.type)
            elif len(out) == 2:
                # Gaussian/Laplace/Student-t models use their location as the
                # centre prediction for post-hoc conformal calibration.
                pred, _ = out
                if self._normalize:
                    pred, label = self._inverse_transform(
                        [pred, label], device=self._device.type
                    )
            else:
                raise RuntimeError(
                    "OD_CQR only supports point output, location-scale "
                    "(loc,scale), or ZINB (n,p,pi) output"
                )
        else:
            pred = out
            if self._normalize:
                pred, label = self._inverse_transform([pred, label], device=self._device.type)
        if pred.ndim != 5:
            raise RuntimeError(f"OD_CQR expects (B,H,O,D,C), got {tuple(pred.shape)}")
        return pred.clamp_min(0), label

    def _collect(self, mode):
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                pred, label = self._point_prediction(X, label)
                preds.append(pred.cpu())
                labels.append(label.cpu())
        return torch.cat(preds), torch.cat(labels)

    def calibrate(self, mode="val"):
        pred, label = self._collect(mode)
        scores = torch.abs(label - pred)
        if self.cqr_mode == "horizon":
            self._conformal_q = torch.stack([
                self._finite_quantile(scores[:, h], self.alpha)
                for h in range(scores.shape[1])
            ])
            msg = ", ".join(f"h{i + 1}={q:.4f}" for i, q in enumerate(self._conformal_q.tolist()))
        else:
            self._conformal_q = self._finite_quantile(scores, self.alpha)
            msg = f"{float(self._conformal_q):.4f}"
        self._logger.info(
            f"OD-CQR calibration on '{mode}' (coverage={1-self.alpha:.1%}, mode={self.cqr_mode}): Q={msg}"
        )

    def _interval(self, pred):
        q = self._conformal_q
        if torch.is_tensor(q) and q.ndim == 1:
            q = q.to(pred).view(1, -1, 1, 1, 1)
        else:
            q = float(q)
        return (pred - q).clamp_min(0), pred + q

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode != "test":
            # Post-hoc CQR is intended for existing checkpoints.  During a
            # train run provide ordinary centre metrics with zero-width bounds.
            pred, label = self._collect(mode)
            target_mode = "valid" if mode == "val" else "test"
            for h in range(pred.shape[1]):
                p, y = self._horizon_slice(pred, h), self._horizon_slice(label, h)
                self.metric.compute_one_batch(p, y, torch.tensor(float("nan")), target_mode, lower=p, upper=p, alpha=self.alpha)
            return
        if not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)
            self.calibrate()
        pred, label = self._collect("test")
        if self._conformal_q is None:
            self.calibrate()
        # Prediction collection stays on CPU to keep the full OD test set
        # memory-safe.  Metric calculation over tens of millions of Chicago
        # OD cells is substantially faster on the configured accelerator.
        pred, label = pred.to(self._device), label.to(self._device)
        lower, upper = self._interval(pred)
        for h in range(pred.shape[1]):
            self.metric.compute_one_batch(
                self._horizon_slice(pred, h), self._horizon_slice(label, h), torch.tensor(float("nan")), "test",
                lower=self._horizon_slice(lower, h), upper=self._horizon_slice(upper, h), alpha=self.alpha,
            )
        if not train_test:
            with self._logger.no_time():
                self._logger.info("\n" + "=" * 25 + "     Test (OD-CQR)     " + "=" * 25)
            for msg in self.metric.get_test_msg():
                self._logger.info(msg)
        if export:
            result = torch.stack([pred, lower, upper, label]).cpu().numpy()
            tag = getattr(self.args, "calibration_tag", "").strip()
            suffix = f"od_cqr_{tag}_res" if tag else "od_cqr_res"
            path = self._get_unique_save_path(suffix)
            np.save(path, result)
            self._logger.info(f"OD-CQR Results Save Path: {path} | Shape: {result.shape}")
