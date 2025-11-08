import torch.nn as nn
from abc import abstractmethod
import torch


class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, seq_len=6, horizon=1):
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


class QuantileRegressor(nn.Module):
    def __init__(self, base_model, in_channels=1, kernel_size=(3,1), padding=(1,0)):
        super().__init__()
        self.base_model = base_model

        self.conv_l = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding)
        self.conv_u = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        X = self.base_model(x)
        y_lower = self.conv_lower(X)
        y_median = self.conv_median(X)
        y_upper = self.conv_upper(X)
        res= torch.cat([y_lower, y_median, y_upper], dim=1)  # [B,3,11,1]
        return res
