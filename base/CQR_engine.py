import numpy as np
import torch

from base.quantile_engine import Quantile_Engine
from base.metrics import compute_all_metrics, masked_mae, masked_kl, masked_crps, masked_mpiw, \
    masked_coverage, masked_wink
from base.metrics import masked_mape
from base.metrics import masked_rmse

class CQR_Engine(Quantile_Engine):
    def __init__(self, **args):
        super(Quantile_Engine, self).__init__(**args)

    def evaluate(self, mode, model_path=None, export=None):
        if mode == 'test' or mode == 'export':
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()

        mids=[]
        lowers=[]
        uppers=[]
        labels = []
        with torch.no_grad():
            mode_=mode
            if mode=='export':
                mode_="test"
            for X, label in self._dataloader[mode_ + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
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

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = masked_mae(mid, label, mask_value).item()
            mape = masked_mape(mid, label, mask_value).item()
            rmse = masked_rmse(mid, label, mask_value).item()
            crps = masked_crps(mid, label, mask_value).item()
            mpiw = masked_mpiw(lower, upper, mask_value).item()
            kl = masked_kl(mid, label, mask_value).item()
            wink = masked_wink(lower, upper, label).item()
            cov = masked_coverage(lower, upper, label).item()
            return mae, mape, rmse, kl, mpiw, crps, wink, cov

        elif mode == 'test' or mode == 'export':
            test_mae = []
            test_mape = []
            test_rmse = []
            test_kl = []
            test_mpiw = []
            test_crps = []
            test_wink = []
            test_cov = []

            self._logger.info(f'check mask value {mask_value}')
            for i in range(1):
                res = compute_all_metrics(mids[:, i, :], labels[:, i, :], mask_value, lowers[:, i, :], uppers[:, i, :])

                # log = ('Test Horizon: {:d}, '
                #        'MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}, WINK: {:.3f}, COV: {:.3f}')
                #
                # self._logger.info(log.format(i + 1, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]))
                test_mae.append(res[0])
                test_rmse.append(res[1])
                test_mape.append(res[2])
                test_kl.append(res[3])
                test_mpiw.append(res[4])
                test_crps.append(res[5])
                test_wink.append(res[6])
                test_cov.append(res[7])

            self._logger.info(f"{self._save_path}")
            log = 'Average Test MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}, KL: {:.3f}, MPIW: {:.3f}, CRPS: {:.3f}, WINK: {:.3f}, COV: {:.3f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape),
                                         np.mean(test_kl), np.mean(test_mpiw), np.mean(test_crps), np.mean(test_wink), np.mean(test_cov)))

            if mode == 'export':
                # mae = torch.mean(test_mae[0].unsqueeze(0), axis=1)
                # mape = torch.mean(test_mape[0].unsqueeze(0), axis=1)
                # rmse = torch.mean(test_rmse[0].unsqueeze(0), axis=1)
                # kl = torch.mean(test_kl[0].unsqueeze(0), axis=1)
                # mpiw = torch.mean(test_mpiw[0].unsqueeze(0), axis=1)
                # crps = torch.from_numpy(test_crps[0])
                # crps = torch.mean(crps.unsqueeze(0), axis=1)
                #
                # wink=torch.mean(test_wink[0].unsqueeze(0), axis=1)
                # cov = torch.mean(test_cov[0].unsqueeze(0), axis=1)
                #
                # metrics = np.vstack((mae, mape, rmse, kl, mpiw, crps, wink, cov))
                # print(metrics.shape)
                # np.save(f"{self._save_path}/metrics.npy", metrics)

                mids.squeeze_(dim=1)
                lowers.squeeze_(dim=1)
                uppers.squeeze_(dim=1)
                labels.squeeze_(dim=1)

                mids.unsqueeze_(dim=0)
                lowers.unsqueeze_(dim=0)
                uppers.unsqueeze_(dim=0)
                labels.unsqueeze_(dim=0)

                result = np.vstack((mids, lowers, uppers, labels))
                self._logger.info(f'export shape (mids, lowers, uppers, labels): {result.shape}')
                np.save(f"{self._save_path}/preds_labels.npy", result)