import torch
from torch import nn
from torch.nn.parameter import Parameter


# this is copied from https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
class ChannelWiseAttention(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super().__init__()
        # first projection layer, bias is True by default 
        self.linear_in = nn.Linear(n_channels, hidden_size)
        # output projection layer, bias is True by default
        self.linear_out = nn.Linear(hidden_size, n_channels)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, x):
        # input size: (batch_size, n_steps, n_channel, n_samples)
        
        # calculate attention map
        # s_shape = (batch_size, n_steps, n_channel)
        s = torch.mean(x, dim = -1)
        # s_shape = (batch_size, n_steps, hidden_size)
        s = self.linear_in(s)
        # s_shape = (batch_size, n_steps, n_channel)
        s = self.tanh(self.linear_out(s))
        attention_map = self.softmax(s)
        # attention_map shape = (batch_size, n_steps, n_channel, 1)
        attention_map = attention_map.unsqueeze(-1)
        # calculate output: cj = aj * xj
        out = x * attention_map
        
        return out
        
if __name__ == "__main__":
    input = torch.randn(2, 32, 64, 64)
    model = eca_layer(channel = 3)

    output = model(input)
    print(output.shape)