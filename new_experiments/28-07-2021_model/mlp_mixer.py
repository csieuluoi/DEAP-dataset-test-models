from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

# def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
#     assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
#     num_patches = (image_size // patch_size) ** 2
#     chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#     return nn.Sequential(
#         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
#         nn.Linear((patch_size ** 2) * channels, dim),
#         *[nn.Sequential(
#             PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
#             PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
#         ) for _ in range(depth)],
#         nn.LayerNorm(dim),
#         Reduce('b n c -> b c', 'mean'),
#         nn.Linear(dim, num_classes)
#     )
def MLPMixer(*, input_length, channels, patch_size, sampling_rate, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (input_length % patch_size) == 0, 'input length must be divisible by patch size'
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    n_patches = (input_length) // (patch_size)

    # input: (_, 32, 8064)
    # rearrange: b= batch, c = n_eeg_channels, n = n_patches/steps, s = sampling_rate, p = patch_size
    return nn.Sequential(
        Rearrange('b c (n p s) -> b n (p s c)', p = patch_size, s = sampling_rate),
        nn.Linear(int(patch_size * sampling_rate * channels), dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(n_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

if __name__ == "__main__":
    model = MLPMixer(
        input_length = 5,
        channels = 32,
        patch_size = 1,
        sampling_rate = 128,
        dim = 512,
        depth = 12,
        num_classes = 2
    )

    img = torch.randn(20, 32, 5*128)
    pred = model(img) # (1, 10
    print(pred)
    # model = MLPMixer(
    #     image_size = 256,
    #     channels = 3,
    #     patch_size = 16,
    #     dim = 512,
    #     depth = 12,
    #     num_classes = 1000
    # )

    # img = torch.randn(1, 3, 256, 256)
    # pred = model(img) # (1, 1000)
