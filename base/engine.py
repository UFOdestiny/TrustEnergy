import os
import time

import numpy as np
import torch

from base.metrics import compute_all_metrics, Metrics


class BaseEngine:
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        sampler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
        args,
        alpha=0.1,
        normalize=True,
        hour_day_month=False,
        metric_list=None,
    ):
        super().__init__()
        self._normalize = normalize
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self.args = args

        self._logger.info("The number of parameters: {}".format(self.model.param_num()))

        self._time_model = "{}_{}.pt".format(
            self.args.model_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        )

        self._logger.info(
            f"Model Save Path: {os.path.join(self._save_path, self._time_model)}"
        )

        # quantile
        self.alpha = alpha
        self.lower_bound = self.alpha / 2
        self.upper_bound = 1 - self.alpha / 2
        self.hour_day_month = hour_day_month

        # metric
        if metric_list is None:
            metric_list = ["MAE", "MAPE", "RMSE", "KL", "CRPS"]
        self.metric = Metrics(self._loss_fn, metric_list, self.model.horizon)

    def split_hour_day_month(self, X, Y):
        data = X[..., 0].unsqueeze(-1)
        hdm = X[..., 1:]
        y = Y[..., 0].unsqueeze(-1)
        return data, hdm, y

    def _predict(self, x, *args):
        return self.model(x)

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            res = []
            for t in tensors:
                if type(t) == tuple:
                    res.append([inv(j) for j in t])
                else:
                    res.append(inv(t))
            return res
        else:
            return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # filename = 'final_model_s{}.pt'.format(self._seed)
        filename = self._time_model
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        # filename = 'final_model_s{}.pt'.format(self._seed)
        filename = self._time_model
        f = os.path.join(save_path, filename)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i[-3:] == ".pt"]
            if len(models) == 0:
                self._logger.info(f"Model {f} Not Exist. No More Models.")
                exit()

            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            m = os.path.join(save_path, models[-1])
            self._logger.info(f"Model {f} Not Exist. Try the Newest Model {m}.")
            f = m

        self.model.load_state_dict(torch.load(f, weights_only=False))

    def load_exact_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False))

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        for X, label in self._dataloader["train_loader"].get_iterator():
            # print(X.shape, label.shape)
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self._predict(X)

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                self._logger.info(f"check mask value {mask_value}")

            scale = None
            if type(pred) == tuple:
                pred, scale = pred  # mean scale

            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            res = self.metric.compute_one_batch(
                pred, label, mask_value, "train", scale=scale
            )
            res.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            self._iter_cnt += 1

    def train(self):
        wait = 0
        min_loss_val = np.inf
        min_loss_test = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            self.evaluate("val")
            v2 = time.time()

            te1 = time.time()
            self.evaluate("test", export=None, train_test=True)
            te2 = time.time()

            valid_loss = self.metric.get_valid_loss()
            test_loss = self.metric.get_test_loss()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(
                epoch + 1, cur_lr, t2 - t1, v2 - v1, te2 - te1
            )
            self._logger.info(msg)

            if test_loss < min_loss_test:
                self._logger.info(
                    "Test loss decrease from {:.3f} to {:.3f}".format(
                        min_loss_test, test_loss
                    )
                )
                min_loss_test = test_loss

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    exit()
                    break

                self.save_model(self._save_path)
                self._logger.info(
                    "Val loss decrease from {:.3f} to {:.3f}".format(
                        min_loss_val, valid_loss
                    )
                )
                min_loss_val = valid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(
                            epoch + 1, min_loss_val
                        )
                    )
                    break

        self.evaluate("test", export=self.args.export)

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and train_test == False:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []
        scales = []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self._predict(X)
                scale = None
                if type(pred) == tuple:
                    pred, scale = pred  # mean scale

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                if scale is not None:
                    scales.append(scale.squeeze(-1).cpu())
        if scales:
            scales = torch.cat(scales, dim=0)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            self.metric.compute_one_batch(
                preds, labels, mask_value, "valid", scale=scale
            )

        elif mode == "test" or mode == "export":
            for i in range(self.model.horizon):
                s = scales[:, i, :].unsqueeze(1) if len(scales) > 0 else None
                self.metric.compute_one_batch(
                    preds[:, i, :].unsqueeze(1),
                    labels[:, i, :].unsqueeze(1),
                    mask_value,
                    "test",
                    scale=s,
                )

            if not train_test:
                for i in self.metric.get_test_msg():
                    self._logger.info(i)

            if export:
                self.save_result(preds, labels)

                # # metrics
                # metrics = np.vstack(self.metric.export())
                # np.save(f"{self._save_path}/metrics.npy", metrics)
                # self._logger.info(f'metrics results shape: {metrics.shape} {self.metric.metric_lst})')

    def save_result(self, preds, labels):
        # results
        # self._logger.info(f"preds/labels shape: {preds.shape}")

        preds.squeeze_(dim=1)
        labels.squeeze_(dim=1)
        preds.unsqueeze_(dim=0)
        labels.unsqueeze_(dim=0)

        result = np.vstack((preds, labels))
        save_name = f"{self.args.model_name}-{self.args.dataset}-res.npy"
        if self.args.result_path:
            path = os.path.join(self.args.result_path, save_name)
        else:
            path = os.path.join(self._save_path, save_name)

        np.save(path, result)
        self._logger.info(f"Results Save Path: {path}")

        self._logger.info(
            f"Results Shape: {result.shape} (preds/labels, timesteps, region)\n\n"
        )
