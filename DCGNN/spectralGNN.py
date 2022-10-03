import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from scipy.sparse import csgraph


# csgraph.laplacian
class SpectralGNN(nn.Module):
    def __init__(self, in_channels, out_channels, K, n_nodes, bias: bool = True):
        super(SpectralGNN, self).__init__()
        self.n_nodes = n_nodes
        # self.register_buffer("W", torch.randn(n_nodes, in_channels), persistent=False)
        # self.W = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes), requires_grad=True)
        # self.W = torch.FloatTensor(n_nodes, in_channels)

        # self.W = torch.nn.Parameter(torch.FloatTensor(n_nodes, 1))
        # self.W.requires_grad = False
        # torch.nn.init.normal_(self.W)

        self.relu = nn.ReLU()
        # self.lin_W = Linear(
        #     in_channels,
        #     n_nodes,
        #     bias=False,  # weight_initializer="glorot"
        # )

        # self.lins = torch.nn.ModuleList(
        #     [
        #         nn.Linear(
        #             in_channels,
        #             out_channels,
        #             bias=False,
        #         )
        #         for _ in range(K)
        #     ]
        # )

        self.lins = torch.nn.ModuleList(
            [
                Linear(
                    in_channels,
                    out_channels,
                    bias=False,  # weight_initializer="glorot"
                )
                for _ in range(K)
            ]
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

    # def calc_degree_matrix_norm(self, a):
    #     return torch.diag(torch.pow(a.sum(dim=-1), -0.5))

    # def create_graph_lapl_norm(self, a):
    #     size = a.shape[-1]
    #     D_norm = self.calc_degree_matrix_norm(a)
    #     # L_norm = torch.ones(size) - (D_norm @ a @ D_norm )
    #     L_norm = D_norm @ (a + torch.ones(size).to(a.device)) @ D_norm
    #     return L_norm

    # def find_eigmax(self, L):
    #     with torch.no_grad():
    #         e1, _ = torch.eig(L, eigenvectors=False)
    #         return torch.max(e1[:, 0]).item()

    # def __norm__(self, W):
    #     # print("W: ", W)

    #     L = self.create_graph_lapl_norm(W)
    #     # print("L: ", L)
    #     # # calculate max eigenvalue

    #     lambda_max = self.find_eigmax(L)

    #     L_star = (2 * L / lambda_max) - torch.eye(self.n_nodes).to(L.device)

    #     return L_star

    def unlaplacian(self, x, L_star):
        out = torch.einsum("c c, b c x -> b c x", L_star, x)
        return out

    def forward(self, x, L_star):
        ## adj matrix is a part of the network and will be learnt during training
        # W_ = self.relu(self.lin_W(self.W)).squeeze()

        # L_star = self.__norm__(W_)
        # print("L_star: ", L_star)

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        # print("out 0:", out)
        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.unlaplacian(x, L_star)
            # print(x, L_star)
            # print(Tx_1)
            out = out + self.lins[1](Tx_1)
            # print("out 1:", out)
        for lin in self.lins[2:]:
            Tx_2 = self.unlaplacian(x, L_star)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2
            # print("out 2:", out)

        if self.bias is not None:
            out += self.bias

        return out


if __name__ == "__main__":
    model = SpectralGNN(4, 8, 5, 32)

    x = torch.randn(2, 32, 4)

    out = model(x)

    print(out)
