import numpy as np
import properscoring as ps
import torch
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import torch.nn.functional as F

zero = torch.tensor(0.0)


class Metrics:
    def __init__(self, loss_func, metric_lst, horizon=1, early_stop_method="MAE"):
        self.dic = {
            "MAE": masked_mae,
            "MSE": masked_mse,
            "MAPE": masked_mape,
            "RMSE": masked_rmse,
            "MPIW": masked_mpiw,
            "CRPS": masked_crps,
            "WINK": masked_wink,
            "COV": masked_coverage,
            "KL": masked_kl,
            "MGAU": mnormal_loss,
            "Quantile": masked_quantile,
        }
        self.horizon = horizon

        self.metric_lst = [loss_func] + metric_lst
        self.metric_func = [self.dic[i] for i in self.metric_lst]
        early_stop_method = loss_func
        self.early_stop_method_index = self.metric_lst.index(early_stop_method)

        self.N = len(self.metric_lst)  # loss function
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]

        self.train_msg = None
        self.test_msg = None
        self.formatter()

    def formatter(self):
        self.train_msg = "Epoch: {:d}, Tr Loss: {:.3f}, "
        for i in self.metric_lst[1:]:
            self.train_msg += "Tr " + i + ": {:.3f}, "
        self.train_msg += "V Loss: {:.3f}, "
        for i in self.metric_lst[1:]:
            self.train_msg += "V " + i + ": {:.3f}, "
        self.train_msg += "Te Loss: {:.3f}, "
        for i in self.metric_lst[1:]:
            self.train_msg += "Te " + i + ": {:.3f}, "

        self.train_msg += (
            "LR: {:.4e}, Tr Time: {:.3f} s/epoch, V Time: {:.3f} s, Te Time: {:.3f} s"
        )

    # quantile=None, upper=None, lower=None
    def compute_one_batch(self, preds, labels, null_val=None, mode="train", **kwargs):
        grad_res = None
        for i, fname in enumerate(self.metric_lst):
            res = None
            if fname in ["MAE", "MSE", "MAPE", "RMSE", "KL", "CRPS"]:
                res = self.metric_func[i](preds, labels, null_val)

            elif fname in ["MGAU"]:
                res = self.metric_func[i](preds, labels, null_val, kwargs["scale"])

            elif fname in ["WINK", "COV"]:
                res = self.metric_func[i](
                    kwargs["lower"], kwargs["upper"], labels, alpha=0.1
                )

            elif fname in ["MPIW"]:
                res = self.metric_func[i](kwargs["lower"], kwargs["upper"])

            elif fname in ["Quantile"]:
                res = self.metric_func[i](
                    kwargs["lower"], preds, kwargs["upper"], labels
                )
            else:
                raise ValueError("Invalid metric name")

            if i == 0 and mode == "train":

                grad_res = res

                # res.backward()  # loss function

            if mode == "train":
                self.train_res[i].append(res.item())
            elif mode == "valid":
                self.valid_res[i].append(res.item())
            else:
                self.test_res[i].append(res.item())

        return grad_res

    def get_loss(self, mode="valid", method="MAE"):
        index_ = self.metric_lst.index(method)

        if mode == "train":
            return self.train_res[index_]
        elif mode == "valid":
            return self.valid_res[index_]
        else:
            return self.test_res[index_]

    def get_valid_loss(self):
        return np.mean(self.valid_res[self.early_stop_method_index])

    def get_test_loss(self):
        return np.mean(self.test_res[self.early_stop_method_index])

    def get_epoch_msg(self, epoch, lr, training_time, valid_time, test_time):
        # print([len(i) for i in self.train_res ])
        # print([len(i) for i in self.valid_res])

        train_lst = [np.mean(i) for i in self.train_res]
        valid_lst = [np.mean(i) for i in self.valid_res]
        test_lst = [np.mean(i) for i in self.test_res]

        msg = self.train_msg.format(
            epoch,
            *train_lst,
            *valid_lst,
            *test_lst,
            lr,
            training_time,
            valid_time,
            test_time,
        )

        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]
        return msg

    def get_test_msg(self):
        msgs = []
        for i in range(self.horizon):
            self.test_msg = f"Test Horizon: {i + 1}, "
            for j in self.metric_lst:
                self.test_msg += j + ": {:.3f}, "
            self.test_msg = self.test_msg[:-2]  # remove the last ", "
            test_lst = [k[i] for k in self.test_res]
            msg = self.test_msg.format(*test_lst)
            msgs.append(msg)

        self.test_msg = f"Average: "
        for i in self.metric_lst:
            self.test_msg += i + ": {:.3f}, "
        self.test_msg = self.test_msg[:-2]
        test_lst = [np.mean(i) for i in self.test_res]
        msg = self.test_msg.format(*test_lst)
        msgs.append(msg)

        self.test_res = [[] for _ in range(self.N)]
        return msgs

    def export(self):
        return self.test_res


def get_mask(labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_pinball(preds, labels, null_val, quantile):
    mask = get_mask(labels, null_val)

    loss = torch.zeros_like(labels, dtype=torch.float)
    error = preds - labels
    smaller_index = error < 0
    bigger_index = 0 < error
    loss[smaller_index] = quantile * (abs(error)[smaller_index])
    loss[bigger_index] = (1 - quantile) * (abs(error)[bigger_index])

    # loss = loss * mask
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mse(preds, labels, null_val):
    # print(preds.shape, labels.shape)
    # Batch size, Horizon, N, Features/N
    assert preds.shape == labels.shape
    # mask = get_mask(labels, null_val)

    loss = (preds - labels) ** 2
    # loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    # print("preds:", preds.shape, "labels:", labels.shape)
    assert preds.shape == labels.shape
    # mask = get_mask(labels, null_val)

    loss = torch.abs(preds - labels)

    # loss = loss * mask
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    mask = get_mask(labels, null_val)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss) * 100  # percent


def masked_kl(preds, labels, null_val):
    # mask = get_mask(labels, null_val)

    loss = labels * torch.log((labels + 1e-5) / (preds + 1e-5))

    # loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_mpiw(lower, upper, null_val=None):
    return torch.mean(upper - lower)


def masked_wink(lower, upper, labels, alpha=0.1):
    score = upper - lower
    score += (2 / alpha) * torch.maximum(lower - labels, zero)
    score += (2 / alpha) * torch.maximum(labels - upper, zero)
    return torch.mean(score)


def masked_coverage(lower, upper, labels, alpha=None):
    in_the_range = torch.sum((labels >= lower) & (labels <= upper))
    coverage = in_the_range / labels.numel() * 100
    return coverage


def masked_nonconf(lower, upper, labels):
    return torch.maximum(lower - labels, labels - upper)


def masked_mpiw_ens(preds, labels, null_val):
    # mask = get_mask(labels, null_val)

    m = torch.mean(preds, dim=list(range(1, preds.dim())))
    # print(torch.min(preds),torch.quantile(m, 0.05),torch.mean(preds),torch.quantile(m, 0.95),torch.max(preds))

    upper_bound = torch.quantile(m, 0.95)
    lower_bound = torch.quantile(m, 0.05)
    loss = upper_bound - lower_bound

    return torch.mean(
        loss
    )  # -torch.mean(torch.quantile(m, 0.8)-torch.quantile(m, 0.2))


def compute_all_metrics(preds, labels, null_val, lower=None, upper=None):
    mae = masked_mae(preds, labels, null_val)
    mape = masked_mape(preds, labels, null_val)
    rmse = masked_rmse(preds, labels, null_val)

    crps = masked_crps(preds, labels, null_val)
    mpiw = masked_mpiw_ens(preds, labels, null_val)
    kl = masked_kl(preds, labels, null_val)

    res = [mae, rmse, mape, kl, mpiw, crps]

    if lower is not None:
        res[4] = masked_mpiw(lower, upper, null_val)
        wink = masked_wink(lower, upper, labels)
        cov = masked_coverage(lower, upper, labels)
        res = res + [wink, cov]

    return res


def nb_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    n, p, pi = preds
    pi = torch.clip(pi, 1e-3, 1 - 1e-3)
    p = torch.clip(p, 1e-3, 1 - 1e-3)

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    lambda_ = 1e-4

    L_yeq0 = torch.log(pi_yeq0 + lambda_) + torch.log(
        lambda_ + (1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0)
    )
    L_yg0 = (
        torch.log(1 - pi_yg0 + lambda_)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0 + lambda_)
        + n_yg0 * torch.log(p_yg0 + lambda_)
        + yg0 * torch.log(1 - p_yg0 + lambda_)
    )

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def nb_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    n, p, pi = preds

    idx_yeq0 = labels <= 0
    idx_yg0 = labels > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = labels[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = labels[idx_yg0]

    index1 = p_yg0 == 1
    p_yg0[index1] = torch.tensor(0.9999)
    index2 = pi_yg0 == 1
    pi_yg0[index2] = torch.tensor(0.9999)
    index3 = pi_yeq0 == 1
    pi_yeq0[index3] = torch.tensor(0.9999)
    index4 = pi_yeq0 == 0
    pi_yeq0[index4] = torch.tensor(0.001)

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = (
        torch.log(1 - pi_yg0)
        + torch.lgamma(n_yg0 + yg0)
        - torch.lgamma(yg0 + 1)
        - torch.lgamma(n_yg0)
        + n_yg0 * torch.log(p_yg0)
        + yg0 * torch.log(1 - p_yg0)
    )

    loss = -torch.sum(L_yeq0) - torch.sum(L_yg0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss)


def gaussian_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    var = torch.pow(scale, 2)
    loss = (labels - loc) ** 2 / var + torch.log(2 * torch.pi * var)

    # pi = torch.acos(torch.zeros(1)).item() * 2
    # loss = 0.5 * (torch.log(2 * torch.pi * var) + (torch.pow(labels - loc, 2) / var))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def laplace_nll_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    loss = torch.log(2 * scale) + torch.abs(labels - loc) / scale

    # d = torch.distributions.poisson.Poisson
    # loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = torch.sum(loss)
    return loss


def mnormal_loss(preds, labels, null_val, scales):
    mask = get_mask(labels, null_val)

    loc, scale = preds, scales

    dis = MultivariateNormal(loc=loc, covariance_matrix=scale)
    loss = dis.log_prob(labels)

    if loss.shape == mask.shape:
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # loss = -torch.sum(loss)
    loss = -torch.mean(loss)
    return loss


def mnb_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    mu, r = preds

    term1 = torch.lgamma(labels + r) - torch.lgamma(r) - torch.lgamma(labels + 1)
    term2 = r * torch.log(r) + labels * torch.log(mu)
    term3 = -(labels + r) * torch.log(r + mu)
    loss = term1 + term2 + term3

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def normal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds
    d = Normal(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def lognormal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    dis = LogNormal(loc, scale)
    loss = dis.log_prob(labels + 0.000001)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def tnormal_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    d = Normal(loc, scale)
    prob0 = d.cdf(torch.Tensor([0]).to(labels.device))
    loss = d.log_prob(labels) - torch.log(1 - prob0)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def laplace_loss(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    loc, scale = preds

    d = Laplace(loc, scale)
    loss = d.log_prob(labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    loss = -torch.sum(loss)
    return loss


def masked_crps(preds, labels, null_val):
    mask = get_mask(labels, null_val)

    # m, v = preds
    # if v.shape != m.shape:
    #     v = torch.diagonal(v, dim1=-2, dim2=-1)

    # loss = ps.crps_gaussian(labels, mu=m, sig=v)
    loss = ps.crps_ensemble(labels.cpu().numpy(), preds.cpu().detach().numpy())

    return loss.mean()


def masked_quantile(
    y_lower,
    y_middle,
    y_upper,
    y_true,
    q_lower=0.05,
    q_upper=0.95,
    q_middle=0.5,
    lam=1.0,
):

    def quantile_loss_(pred, target, quantile):
        error = target - pred
        return torch.max((quantile - 1) * error, quantile * error).mean()

    def monotonicity_loss(y_lower, y_middle, y_upper, margin=0.0):
        loss = F.relu(y_lower - y_middle + margin) + F.relu(y_middle - y_upper + margin)
        return loss.mean()

    loss_lower = quantile_loss_(y_lower, y_true, q_lower)
    loss_middle = quantile_loss_(y_middle, y_true, q_middle)
    loss_upper = quantile_loss_(y_upper, y_true, q_upper)
    loss_monotonic = monotonicity_loss(y_lower, y_middle, y_upper)
    return loss_lower + loss_middle + loss_upper + lam * loss_monotonic


if __name__ == "__main__":
    f = "Epoch: {:d}, T Loss: {:.3f},T Loss: {:.3f},T Loss: {:.3f}"
    print(f.format(1, *[1, 1, 1]))
