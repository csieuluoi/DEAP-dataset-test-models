import torch
from torch import nn
from torch.nn import functional as F
import torch.fft
from einops import repeat

class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    # return torch.fft.fft2(x, dim=(-1, -2)).real
    return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real

class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.TransformerEncoder):
    def __init__(
        self, d_model=256, expansion_factor=2, dropout=0.5, num_layers=6, n_classes = 1
    ):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, n_classes)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, return_features = False):
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # print(cls_tokens.size())
        # x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            x = layer(x)

        x = x[:, 0]
        if return_features:
            return x
        return self.out(x)
