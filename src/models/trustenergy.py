import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
import numpy as np

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

class Norm_T(nn.Module):
    def __init__(self, c_in, c_out, seq_len, out_dim):
        super(Norm_T, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(seq_len-out_dim + 1, 1),
                                bias=True)

    def forward(self, x):  # B, Horizon, N, F
        x = x.permute(0, 2, 1, 3)
        y = self.n_conv(x)
        y = F.softplus(y)
        y = y.permute(0, 2, 1, 3)
        return y

class Norm_S(nn.Module):
    def __init__(self, c_in, c_out, out_dim):
        super(Norm_S, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out+out_dim-1,
                                kernel_size=(1, 1),
                                bias=True)

    def forward(self, x):  # B, Horizon, N, F
        y = self.n_conv(x)
        y = F.softplus(y)
        return y

class DGCN(nn.Module):
    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(DGCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h, embd=None):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size, input_size, num_node, feature = X.shape
        # batch_size = X.shape[0]  # batch_size
        # num_node = X.shape[1]
        # input_size = X.size(2)  # time_length
        # # feature = X.shape[3]

        supports = [A_q, A_h]

        x0 = X.permute(3, 2, 1, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[feature, num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        for support in supports:
            s = support.unsqueeze(0).expand(x0.shape[0], -1, -1)
            x1 = torch.matmul(s, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(s, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, feature, num_node, input_size, batch_size])
        x = x.permute(1, 4, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[feature, batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        x = x.permute(1, 3, 2, 0)
        return x

class MetaDGCN(nn.Module):
    def __init__(self, in_channels, out_channels, orders, time_dim, activation='relu'):
        """
        Meta-learning enhanced Diffusion GCN (MetaDGCN)。

        :param in_channels: 输入通道数（在本代码中相当于时间步数，且后续要求 in_channels == feature 数）
        :param out_channels: 每个节点输出的特征数
        :param orders: 扩散的阶数，模型中将生成 2 * orders + 1 个扩散矩阵
        :param time_dim: 时间 embedding 的维度（全局，每个 batch 一个向量）
        :param activation: 激活函数
        """
        super(MetaDGCN, self).__init__()
        self.in_channels = in_channels  # 与 X 中最后一个维度的值对应
        self.out_channels = out_channels  # 与 X 中最后一个维度的值对应
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.time_dim = time_dim

        # 原始的传播权重 Theta1，形状为 [in_channels * num_matrices, out_channels]
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        # Meta-Network：输入时间 embedding（全局，仅 [batch_size, time_dim]），输出 2 * (in_channels*num_matrices*out_channels)
        self.meta_net = nn.Linear(time_dim, in_channels * self.num_matrices * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        # 将 x_ 增加一个最外层维度后，与 x 在第0维拼接
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h, time_emb):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size, input_size, num_node, feature = X.shape
        # batch_size = X.shape[0]  # batch_size
        # num_node = X.shape[1]
        # input_size = X.size(2)  # time_length
        # # feature = X.shape[3]

        supports = [A_q, A_h]

        x0 = X.permute(3, 2, 1, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[feature, num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        for support in supports:
            s = support.unsqueeze(0).expand(x0.shape[0], -1, -1)
            x1 = torch.matmul(s, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(s, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        delta = self.meta_net(time_emb)
        delta = delta.view(batch_size, self.in_channels * self.num_matrices, self.out_channels)
        Theta1_exp = self.Theta1.unsqueeze(0).expand(batch_size, -1, -1)
        effective_Theta = Theta1_exp + delta  # (B, T*num_matrices, out_channels)


        x = torch.reshape(x, shape=[self.num_matrices, feature, num_node, input_size, batch_size])
        x = x.permute(1, 4, 2, 3, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])

        x = torch.bmm(x, effective_Theta).unsqueeze(0)
        # x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)

        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        x = x.permute(1, 3, 2, 0)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', device='cuda'):
        super(TCN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        # batch_size = X.shape[0]
        # seq_len = X.shape[1]
        # Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        batch_size, seq_len, num_nodes, num_features = X.shape

        Xf = X
        inv_idx = torch.arange(Xf.size(1) - 1, -1, -1).long().to(
            device=self.device)  # .to(device=self.device).to(device=self.device)
        Xb = Xf.index_select(1, inv_idx)  # inverse the direction of time

        Xf = Xf.permute(0, 2, 3, 1)
        Xb = Xb.permute(0, 2, 3, 1)  # (batch_size, num_nodes, 1, num_timesteps)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features])

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels, num_features])

        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels, num_features]).to(
            device=self.device)  # .to(device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)  # (batch_size, num_timesteps, out_features)

        inv_idx = torch.arange(outb.size(1) - 1, -1, -1).long().to(device=self.device)  # .to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)
        return out

class HieGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                # print(embeddings.shape, self.weights_pool.shape)
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv

class HieGCRU(nn.Module):
    def __init__(
        self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HieGCN(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HieGCN(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x, state, support, embeddings):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))

        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)

class HieEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="S",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HieGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HieGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        in_steps = x.shape[1]

        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(in_steps):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)

        # current_input: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_input, output_hidden

class HieDecoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="ST",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HieGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HieGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, xt, init_state, support, embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden

class TrustE(BaseModel):
    def __init__(self, A, node_num, hidden_dim_t, hidden_dim_s, rank_t, rank_s,
                 num_timesteps_input, num_timesteps_output, device, input_dim, output_dim, seq_len,
                 node_embedding_dim=16, embedding_dim = 12, **args):
        super(TrustE, self).__init__(node_num, input_dim, output_dim, **args)
        self.node_num=node_num
        self.num_feature = input_dim
        self.seq_len = seq_len
        self.num_layers=1

        self.hour_embedding = nn.Embedding(24, embedding_dim)
        self.day_embedding = nn.Embedding(7, embedding_dim)
        self.month_embedding = nn.Embedding(12, embedding_dim)
        self.node_embedding = nn.init.xavier_normal_(nn.Parameter(torch.empty(node_num, node_embedding_dim)))

        self.st_proj = nn.Linear(64, embedding_dim)
        self.out_proj = nn.Linear(64, output_dim)

        self.encoder_t = HieEncoder(node_num, input_dim, 64, 2, 1, embedding_dim * 3, meta_axis="T")
        self.encoder_s = HieEncoder(node_num, input_dim, 64, 2, 1,
            node_embedding_dim, meta_axis="S", # meta_axis=None,
        )
        self.decoder = HieDecoder(node_num, output_dim, 64, 2, 1, embedding_dim, )


        # self.TC1 = TCN(node_num, hidden_dim_t, kernel_size=3).to(device=device)
        # self.TC2 = TCN(hidden_dim_t, rank_t, kernel_size=3, activation='linear').to(device=device)
        # self.TC3 = TCN(rank_t, hidden_dim_t, kernel_size=3).to(device=device)
        # self.TGau = Norm_T(hidden_dim_t, node_num, self.seq_len, output_dim).to(device=device)

        # self.SC1 = DGCN(num_timesteps_input, hidden_dim_s, 3).to(device=device)
        # self.SC2 = DGCN(hidden_dim_s, rank_s, 2, activation='linear').to(device=device)
        # self.SC3 = DGCN(rank_s, hidden_dim_s, 2).to(device=device)

        self.SC1 = MetaDGCN(num_timesteps_input, hidden_dim_s, 3,embedding_dim * 3).to(device=device)
        self.SC2 = MetaDGCN(hidden_dim_s, rank_s, 2,embedding_dim * 3, activation='linear').to(device=device)
        self.SC3 = MetaDGCN(rank_s, hidden_dim_s, 2,embedding_dim * 3).to(device=device)

        self.SGau = Norm_S(hidden_dim_s, num_timesteps_output, self.num_feature).to(device=device)

        self.A = A
        self.A_q = torch.from_numpy(calculate_random_walk_matrix(self.A).T.astype('float32'))
        self.A_h = torch.from_numpy(calculate_random_walk_matrix(self.A.T).T.astype('float32'))
        self.A_q = self.A_q.to(device=device)
        self.A_h = self.A_h.to(device=device)

        # self.Attention=Attention()

    def forward(self, X): #, HDM
        # batch_size, input_len, N, feature
        XX=X
        X=XX[...,[0]]
        HDM=XX[...,1:]

        x_t=0
        x_s=0

        hour_emb = self.hour_embedding(HDM[:,-1,0,0].int())
        day_emb = self.day_embedding(HDM[:,-1,0,1].int())
        month_emb = self.month_embedding(HDM[:,-1,0,2].int())
        time_embedding = torch.cat([hour_emb, day_emb, month_emb], dim=-1) #[64, 48]

        # support = torch.softmax(torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1 )
        #
        # h_s, _ = self.encoder_s(X, support, self.node_embedding)
        # h_t, _ = self.encoder_t(X, support, time_embedding)
        # h_last=(h_s + h_t)[:, -1, :, :]
        #
        # st_embedding = self.st_proj(h_last)  # B, N, st_emb_dim
        # support = torch.softmax(torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)), dim=-1, )
        # ht_list = [h_last] * self.num_layers
        # go = torch.zeros((X.shape[0], self.node_num, self.output_dim), device=X.device)
        # out = []
        # for t in range(1):
        #     h_de, ht_list = self.decoder(go, ht_list, support, st_embedding)
        #     go = self.out_proj(h_de)
        #     out.append(go)
        #
        # x_t = torch.stack(out, dim=1).permute(0, 3, 2,1)

        # X_t1 = self.TC1(X)
        # X_t2 = self.TC2(X_t1)
        # X_t3 = self.TC3(X_t2)
        # x_t = self.TGau(X_t3)

        # X=X[:,:,:,0].permute(0,2,1)
        X_s1 = self.SC1(X, self.A_q, self.A_h, time_embedding)
        X_s2 = self.SC2(X_s1, self.A_q, self.A_h, time_embedding)
        X_s3 = self.SC3(X_s2, self.A_q, self.A_h, time_embedding)
        x_s = self.SGau(X_s3)

        y=x_t+x_s
        # print(y.shape)
        return y

