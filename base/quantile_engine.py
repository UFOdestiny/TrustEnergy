import time
import numpy as np
import torch
import os

from base.engine import BaseEngine
from base.metrics import (
    compute_all_metrics,
    masked_mae,
    masked_kl,
    masked_crps,
    masked_mpiw,
    masked_coverage,
    masked_wink,
    masked_nonconf,
    Metrics,
    masked_quantile,
)
from base.metrics import masked_mape
from base.metrics import masked_rmse


class Quantile_Engine(BaseEngine):
    def __init__(self, **args):
        super(Quantile_Engine, self).__init__(**args)
        # if args["metric_list"] is None:
        args["metric_list"] = [
            "Quantile",
            "MAE",
            "MAPE",
            "RMSE",
            "KL",
            "CRPS",
            "MPIW",
            "WINK",
            "COV",
        ]
        self._loss_fn = "Quantile"

        self.metric = Metrics(
            self._loss_fn, args["metric_list"], 1
        )  # self.model.horizon

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()
            X, label = self._to_device(self._to_tensor([X, label]))

            if self.hour_day_month:
                X, hdm, label = self.split_hour_day_month(X, label)
                pred = self.model(X, hdm)
            else:
                pred = self._predict(X)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f"check mask value {mask_value}")

            if type(pred) == tuple:
                pred, _ = pred
            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            # print(pred.shape)
            # print(self.metric.metric_lst)
            mid = torch.unsqueeze(pred[:, 0, :, :], 1)
            lower = torch.unsqueeze(pred[:, 1, :, :], 1)
            upper = torch.unsqueeze(pred[:, 2, :, :], 1)

            res = self.metric.compute_one_batch(
                mid, label, mask_value, "train", upper=upper, lower=lower
            )
            res.backward()

            # loss=masked_quantile(lower, mid, upper, label,self.lower_bound,self.upper_bound)
            # loss.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and train_test == False:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        mids = []
        lowers = []
        uppers = []

        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                if self.hour_day_month:
                    X, hdm, label = self.split_hour_day_month(X, label)
                    pred = self.model(X, hdm)
                else:
                    pred = self._predict(X)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                mid = torch.unsqueeze(pred[:, 0, :, :], 1)
                lower = torch.unsqueeze(pred[:, 1, :, :], 1)
                upper = torch.unsqueeze(pred[:, 2, :, :], 1)

                mids.append(mid.squeeze(-1).cpu())
                lowers.append(lower.squeeze(-1).cpu())
                uppers.append(upper.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            self.metric.compute_one_batch(
                mids, labels, mask_value, "valid", upper=uppers, lower=lowers
            )

        elif mode == "test" or mode == "export":
            # for i in range(self.model.horizon):
            self.metric.compute_one_batch(
                mids,
                labels,
                mask_value,
                "test",
                upper=uppers,
                lower=lowers,
            )

            if not train_test:
                for i in self.metric.get_test_msg():
                    self._logger.info(i)

            if export:
                self.save_result(mids, uppers, lowers, labels)

    def save_result(self, mids, uppers, lowers, labels):
        mids.squeeze_(dim=1)
        lowers.squeeze_(dim=1)
        uppers.squeeze_(dim=1)
        labels.squeeze_(dim=1)

        mids.unsqueeze_(dim=0)
        lowers.unsqueeze_(dim=0)
        uppers.unsqueeze_(dim=0)
        labels.unsqueeze_(dim=0)

        result = np.vstack((mids, lowers, uppers, labels))
        save_name = f"{self.args.model_name}-{self.args.dataset}-res.npy"
        if self.args.result_path:
            path = os.path.join(self.args.result_path, save_name)
        else:
            path = os.path.join(self._save_path, save_name)

        np.save(path, result)
        self._logger.info(f"Results Save Path: {path}")

        self._logger.info(
            f"Results Shape: {result.shape} (mids/lowers/uppers/labels, timesteps, region)\n\n"
        )

    def cqr(self):
        mids = []
        lowers = []
        uppers = []
        labels = []

        with torch.no_grad():
            for X, label in self._dataloader["val_loader"].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)

                if type(pred) == tuple:
                    pred, _ = pred

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                mid = torch.unsqueeze(pred[:, 0, :, :], 1)
                lower = torch.unsqueeze(pred[:, 1, :, :], 1)
                upper = torch.unsqueeze(pred[:, 2, :, :], 1)

                mids.append(mid.squeeze(-1).cpu())
                lowers.append(lower.squeeze(-1).cpu())
                uppers.append(upper.squeeze(-1).cpu())

                labels.append(label.squeeze(-1).cpu())

        mids = torch.cat(mids, dim=0)
        lowers = torch.cat(lowers, dim=0)
        uppers = torch.cat(uppers, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        nonconf_set = masked_nonconf(lowers, uppers, labels)
        bound = torch.quantile(nonconf_set, (1 - self.alpha) * (1 + 1), dim=0)
