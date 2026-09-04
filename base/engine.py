import os
import time

import numpy as np
import torch
from base.metrics import Metrics


class BaseEngine:
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_norm,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
        args,
        normalize=True,
        metric_list=None,
        init_weights=False,
        **kwargs,
    ):
        super().__init__()

        self._normalize = normalize
        self._device = device
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_norm = clip_grad_norm
        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self.args = args
        self._mask_value = torch.tensor(float("nan"))

        self.model = model
        self.model.to(self._device)

        # Optional Xavier weight initialization (replaces AGCRN/ASTGCN/DSTAGNN engines)
        if init_weights:
            self._init_model_weights()

        # Metrics
        if metric_list is None:
            metric_list = ["MAE", "MAPE", "MSE", "RMSE", "KL", "CRPS"]
        self.metric = Metrics(self._loss_fn, metric_list, self.model.horizon)

        self._logger.info(f"{'Loss Function':20s}: {self._loss_fn}")
        self._logger.info(f"{'Parameters':20s}: {self.model.param_num()}")

        self._time_model = "{}_{}.pt".format(
            self.args.model_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        )

        self._logger.info(
            f"Model Save Path: {os.path.join(self._save_path, self._time_model)}"
        )

    def _init_model_weights(self):
        """Xavier/uniform initialization for model parameters."""
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def _predict(self, x, label, iter, *args):
        return self.model(x)

    def _collect(self, t):
        """Post-process a prediction/label tensor before stacking over the test
        set.  Flow models drop the trailing singleton feature axis so tensors
        are ``(B, T, N)``; OD engines override this to keep the OD-matrix axes.
        """
        return t.squeeze(-1)

    def _horizon_slice(self, t, i):
        """Slice horizon step *i* from a stacked tensor, keeping a length-1 time
        axis at position 1 so the metric sees ``(B, 1, ...)``.  Default assumes
        the flow layout ``(B, T, N)``; OD engines override for ``(B, T, N, N, D)``.
        """
        return t[:, i, :].unsqueeze(1)

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [t.to(self._device) for t in tensors]
        return tensors.to(self._device)

    def _prepare_batch(self, batch):
        return self._to_device(self._to_tensor(batch))

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [
                arr if isinstance(arr, torch.Tensor)
                else torch.tensor(arr, dtype=torch.float32)
                for arr in nparray
            ]
        if isinstance(nparray, torch.Tensor):
            return nparray
        return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors, device="cuda"):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor, device=device)

        if isinstance(tensors, list):
            res = []
            for t in tensors:
                if isinstance(t, tuple):
                    res.append([inv(j) for j in t])
                else:
                    res.append(inv(t))
            return res
        return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = self._time_model
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

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

        self.model.load_state_dict(torch.load(f, weights_only=False))

    def load_exact_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False))

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

            # X (b, t, n, f), label (b, t, n, f)
            X, label = self._prepare_batch([X, label])
            pred = self._predict(X, label=label, iter=self._iter_cnt)

            scale = None
            if isinstance(pred, tuple):
                pred, scale = pred

            if self._normalize:
                pred, label = self._inverse_transform(
                    [pred, label], device=self._device.type
                )

            res = self.metric.compute_one_batch(
                pred, label, mask_value, "train", scale=scale
            )

            # A single non-finite loss (e.g. an SSM gradient blow-up late in
            # training) would otherwise poison every weight: clip_grad_norm_
            # turns a NaN gradient into a NaN scale, so clipping cannot save it
            # and optimizer.step() spreads the NaN, after which all subsequent
            # predictions are NaN — the masked metrics report 0.000 and the
            # train loop aborts with "Something went WRONG!".  Skip the step.
            if not torch.isfinite(res):
                self._optimizer.zero_grad()
                self._iter_cnt += 1
                continue

            res.backward()

            if self._clip_grad_norm != 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_norm
                )
                # Non-finite gradients can arise even from a finite loss; a NaN
                # norm makes clipping a no-op, so skip the step rather than
                # corrupt the weights (see the loss guard above).
                if not torch.isfinite(total_norm):
                    self._optimizer.zero_grad()
                    self._iter_cnt += 1
                    continue
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
                    "Test loss: {:.3f} -> {:.3f}".format(min_loss_test, test_loss)
                )
                min_loss_test = test_loss

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    exit()
                    break

                self.save_model(self._save_path)
                self._logger.info(
                    "Val  loss: {:.3f} -> {:.3f}".format(min_loss_val, valid_loss)
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
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds, labels, scales = [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                # print(X.shape, label.shape)
                pred = self._predict(X, label=label, iter=self._iter_cnt)
                scale = None
                if isinstance(pred, tuple):
                    pred, scale = pred

                if self._normalize:
                    pred, label = self._inverse_transform(
                        [pred, label], device=self._device.type
                    )

                if mode == "val":
                    self.metric.compute_one_batch(
                        pred,
                        label,
                        self._mask_value.to(pred.device),
                        "valid",
                        scale=scale,
                    )
                else:
                    preds.append(self._collect(pred).cpu())
                    labels.append(self._collect(label).cpu())
                    if scale is not None:
                        scales.append(self._collect(scale).cpu())

        if mode == "val":
            return

        scales = torch.cat(scales, dim=0) if scales else None
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                s = self._horizon_slice(scales, i) if scales is not None else None
                self.metric.compute_one_batch(
                    self._horizon_slice(preds, i),
                    self._horizon_slice(labels, i),
                    mask_value,
                    "test",
                    scale=s,
                )

            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

            if export:
                self.save_result(preds, labels)
                self.save_test()

    def save_result(self, preds, labels):
        # preds: (B, T, N) or (B, T, N, F)
        # labels: (B, T, N) or (B, T, N, F)
        preds = preds.unsqueeze(0)
        labels = labels.unsqueeze(0)
        # -> (2, B, T, N, ...)
        result = torch.cat([preds, labels], dim=0)
        result_np = result.cpu().numpy()

        path = self._get_unique_save_path("res")

        np.save(path, result_np)

        self._logger.info(f"Results Save Path: {path}")
        self._logger.info(
            f"Results Shape: {result_np.shape} (preds/labels, test size, horizon, region, channels)\n\n"
        )

    def _get_unique_save_path(self, suffix):
        base_name = f"{self.args.model_name}-{self.args.dataset}-{suffix}"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)

        index = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{index}.npy"
            path = os.path.join(self._save_path, save_name)
            index += 1

        return path

    def save_test(self):
        test_data = []

        with torch.no_grad():
            for X, _ in self._dataloader["test_loader"].get_iterator():
                t = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
                test_data.append(t.cpu())

        if not test_data:
            self._logger.info("Test data is empty. Skip saving test inputs.")
            return

        test_tensor = torch.cat(test_data, dim=0)
        path = self._get_unique_save_path("test")
        test_np = test_tensor.numpy()
        np.save(path, test_np)

        self._logger.info(f"Test Save Path: {path}")
        self._logger.info(
            f"Test Shape: {test_np.shape} (total test size, seq_len, region, feature)\n\n"
        )


class BaseEngine_OD(BaseEngine):
    """Engine for origin-destination models (see :class:`base.model.BaseODModel`).

    OD predictions and labels are 5-D ``(B, horizon, N, N, D)`` — an ``N×N``
    matrix with ``D`` mobility channels per forecast step — whereas the flow
    pipeline assumes ``(B, T, N, feature=1)`` and drops the trailing axis.
    ``BaseEngine`` already routes the two shape-dependent steps through
    :meth:`_collect` and :meth:`_horizon_slice`; this subclass overrides only
    those, so all of the training loop, checkpointing, logging, metric tracking
    and export are inherited unchanged.
    """

    def _collect(self, t):
        # Keep the full OD-matrix layout (B, H, N, N, D); do not squeeze.
        return t

    def _horizon_slice(self, t, i):
        # (B, H, N, N, D) -> (B, 1, N, N, D) for horizon step i.
        return t[:, i : i + 1]


class BaseEngine_OD_Stat(BaseEngine_OD):
    """Base engine for non-neural *statistical* OD models (ARIMA, SARIMA, VAR,
    HA).

    These models are not trained by gradient descent: they are fit and forecast
    in a single call over plain numpy arrays.  The dataloader (``load_dataset_plain``)
    hands back ``(train, valid, test)``, each ``(T, N, N, D)``.  A subclass
    implements :meth:`_forecast` returning a prediction array shaped like
    ``test``; this base handles inverse-transform, metric reporting, and export.

    Masking uses ``NaN`` (i.e. *every* OD cell counts, including the many true
    zeros) — identical to the neural OD engine, so statistical and neural
    baselines are measured on the same footing.
    """

    def _forecast(self, train, valid, test):
        """Fit on *train* and forecast ``len(test)`` steps ahead.  Override for
        a different protocol (HA forecasts from the validation tail)."""
        return self.model(train, test.shape[0])

    def train(self, export=False):
        train, valid, test = self._dataloader

        pred = np.asarray(self._forecast(train, valid, test), dtype=np.float32)
        pred = torch.from_numpy(pred)
        test_t = torch.from_numpy(np.asarray(test, dtype=np.float32))

        if self._normalize:
            pred, test_t = self._inverse_transform([pred, test_t], device="cpu")

        # Whole test set scored as one batch (no sliding window for these
        # baselines); NaN mask => all cells counted, matching the neural engine.
        mask_value = torch.tensor(float("nan"))
        self.metric.compute_one_batch(pred, test_t, mask_value, "test")

        with self._logger.no_time():
            self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
        for msg in self.metric.get_test_msg():
            self._logger.info(msg)

        if export:
            self.save_result(pred, test_t)