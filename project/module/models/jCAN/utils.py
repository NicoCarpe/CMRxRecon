import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F

def fft2d(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, H, W)
    return torch.fft.fft2(x, dim=(-2, -1), norm='ortho')

def ifft2d(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, H, W)
    return torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')

def fftshift2d(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, H, W)
    return torch.roll(x, shifts=(x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))

def ifftshift2d(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, H, W)
    return torch.roll(x, shifts=((x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2), dims=(-2, -1))

def rss(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, H, W)
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

def rss2d(x):
    assert len(x.shape) == 2
    return (x.real ** 2 + x.imag ** 2).sqrt()

def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ux = F.conv2d(X, w)
    uy = F.conv2d(Y, w)
    uxx = F.conv2d(X * X, w)
    uyy = F.conv2d(Y * Y, w)
    uxy = F.conv2d(X * Y, w)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux ** 2 + uy ** 2 + C1,
        vx + vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D
    return 1 - S.mean()

def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=1)

def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    _, c, _, _, _ = x.shape
    assert c % 2 == 0
    c = c // 2
    return torch.complex(x[:, :c], x[:, c:])

def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fft2d(image * sens_maps)

def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (ifft2d(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
