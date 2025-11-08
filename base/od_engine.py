import time
import numpy as np
import torch

from base.engine import BaseEngine


class OD_Engine(BaseEngine):
    def __init__(self, **args):
        super(OD_Engine, self).__init__(**args)

    def train_batch(self):
        self.model.train()
        self._dataloader['train_loader'].shuffle()
        for X, label in self._dataloader['train_loader'].get_iterator():
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
                self._logger.info(f'check mask value {mask_value}')

            scale = None
            if type(pred) == tuple:
                pred, scale = pred  # mean scale

            if self._normalize:
                pred, label = self._inverse_transform([pred, label])

            self.metric.compute_one_batch(pred, label, mask_value, "train", scale=scale)

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            self._iter_cnt += 1

    def train(self):
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            self.evaluate('val')
            v2 = time.time()
            valid_loss = self.metric.get_valid_loss()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(epoch + 1, cur_lr, t2 - t1, v2 - v1)
            self._logger.info(msg)

            if valid_loss < min_loss:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    break

                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.3f} to {:.3f}'.format(min_loss, valid_loss))
                min_loss = valid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')

    def evaluate(self, mode, model_path=None, export=None):
        if mode == 'test':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []
        scales = []

        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)

                scale = None
                if type(pred) == tuple:
                    pred, scale = pred  # mean scale

                if self._normalize:
                    pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                if scale:
                    scales.append(scale.squeeze(-1).cpu())
        if scales:
            scales = torch.cat(scales, dim=0)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            self.metric.compute_one_batch(pred, label, mask_value, "valid", scale=scale)

        elif mode == 'test':
            self._logger.info(f'check mask value {mask_value}')

            for i in range(self.model.horizon):
                s = scales[:, i, :] if len(scales) > 0 else None
                self.metric.compute_one_batch(preds[:, i, :], labels[:, i, :], mask_value, "test", scale=s)

            for i in self.metric.get_test_msg():
                self._logger.info(i)

            if export:
                # results
                preds.squeeze_(dim=1)
                labels.squeeze_(dim=1)
                preds.unsqueeze_(dim=0)
                labels.unsqueeze_(dim=0)
                result = np.vstack((preds, labels))
                np.save(f"{self._save_path}/preds_labels.npy", result)
                self._logger.info(f'prediction results shape: {result.shape} (preds/labels, timesteps, region)')

                # # metrics
                # metrics = np.vstack(self.metric.export())
                # np.save(f"{self._save_path}/metrics.npy", metrics)
                # self._logger.info(f'metrics results shape: {metrics.shape} {self.metric.metric_lst})')
