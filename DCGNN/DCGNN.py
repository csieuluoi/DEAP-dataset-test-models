import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops, get_laplacian, dense_to_sparse
from torch_geometric.transforms import LaplacianLambdaMax


class DCGNN(nn.Module):
    def __init__(
        self, in_features, out_features, n_channels, K, n_classes=2, dropout_rate=0.0
    ):
        super(DCGNN, self).__init__()
        self.n_channels = n_channels
        # self.W_star = nn.Parameter(
        #     torch.FloatTensor(n_channels, n_channels), requires_grad=True
        # )
        coord = torch.randn(n_channels, n_channels, 4)

        nn.init.xavier_uniform_(coord)
        self.adj_fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.ReLU()
        )
        self.register_buffer("coord", coord)
        self.GC_layer = tgnn.ChebConv(
            in_channels=in_features,
            out_channels=out_features,
            K=K,
            normalization="sym",
        )

        # self.GC_layer1 = tgnn.ChebConv(
        #     in_channels=out_features,
        #     out_channels=out_features,
        #     K=K,
        #     normalization="sym",
        # )

        # self.GC_layer2 = tgnn.ChebConv(
        #     in_channels=out_features,
        #     out_channels=out_features,
        #     K=K,
        #     normalization="sym",
        # )

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        # self.get_lambda_max = LaplacianLambdaMax(normalization="sym")
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_features * n_channels, n_classes),
        )

    def forward(self, x):

        """I'm strugling with trainable adjacency matrix!!! the weight W_star is not trained!"""
        self.W_star = self.adj_fc(self.coord).squeeze()
        # edge_index, edge_weight = dense_to_sparse(self.W_star)
        edge_index = (self.W_star > 0.1).nonzero().t()
        # print(len(edge_index))
        # print(edge_index)
        edge_weight = self.W_star[edge_index[0], edge_index[1]]

        x = self.GC_layer(x, edge_index, edge_weight)
        # x = self.GC_layer1(x, edge_index, edge_weight)
        # x = self.GC_layer2(x, edge_index, edge_weight)

        x = x.unsqueeze(1)
        x = self.conv11(x)
        x = x.reshape(x.shape[0], -1)
        # x = torch.mean(x, dim=1)
        # x = x[:, 0]
        out = self.fc(x)

        return out


if __name__ == "__main__":
    model = DCGNN(4, 8, 32, 5, 2, 0.2)
    x = torch.randn(2, 32, 4)

    out = model(x)
    print(out.shape)
