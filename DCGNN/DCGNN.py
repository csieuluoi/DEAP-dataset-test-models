from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, get_laplacian, dense_to_sparse
from torch_geometric.transforms import LaplacianLambdaMax

# from spectralGNN import SpectralGNN
from GraphCN import GraphCN


def device_as(x, y):
    return x.to(y.device)


# tensor operations now support batched inputs
def calc_degree_matrix_norm(a):
    return torch.diag_embed(torch.pow(a.sum(dim=-1), -0.5))


def create_graph_lapl_norm(a):
    size = a.shape[-1]
    a += device_as(torch.eye(size), a)
    D_norm = calc_degree_matrix_norm(a)
    L_norm = torch.bmm(torch.bmm(D_norm, a), D_norm)
    return L_norm


class DCGNN(nn.Module):
    def __init__(
        self, in_features, out_features, n_channels, K, n_classes=2, dropout_rate=0.0
    ):
        super(DCGNN, self).__init__()
        self.n_channels = n_channels

        coord = torch.randn(n_channels, n_channels, 4)

        self.adj_fc = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU()
        )
        self.register_buffer("coord", coord)
        # self.GC_layer = tgnn.ChebConv(
        #     in_channels=in_features,
        #     out_channels=out_features,
        #     K=K,
        #     normalization="sym",
        # )

        self.GC_layer = GraphCN(in_features=in_features, out_features=out_features)
        self.GC_layer1 = GraphCN(in_features=out_features, out_features=out_features)

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        # self.get_lambda_max = LaplacianLambdaMax(normalization="sym")
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(in_features * n_channels, out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_features, n_classes),
        )

    def forward(self, x):

        adj = self.adj_fc(self.coord).squeeze()
        # print(adj.shape)
        print(adj)

        ## test ChebConv module in pytorch geometric
        # edge_index = (adj > 0).nonzero().t()
        # edge_weight = adj[edge_index[0], edge_index[1]]

        # b, n, _ = x.shape
        # x = x.reshape(b * n, -1)
        # x = self.GC_layer(x, edge_index=edge_index, edge_weight=edge_weight)
        # x = self.relu(x)

        # x = x.reshape(b, n, -1)

        ## test GraphCN
        x = self.GC_layer(x, adj)
        x = self.GC_layer1(x, adj)
        x = x.unsqueeze(1)
        x = self.conv11(x)
        # print(torch.mean(x.squeeze(), dim=1).shape)
        # # x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = torch.mean(x.squeeze(), dim=1)
        x = x[:, 0]

        out = self.fc(x)

        return out


if __name__ == "__main__":
    model = DCGNN(4, 8, 32, 5, 2, 0.2)
    x = torch.randn(2, 32, 4)

    out = model(x)
    print(out)
