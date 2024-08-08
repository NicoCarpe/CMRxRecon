"""
@author: sunkg

Jul 23, 2024
Extended by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""

import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F

def fft2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, T, C, H, W)
    return torch.fft.fft2(x, dim=(-2, -1), norm='ortho')

def ifft2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, T, C, H, W)
    return torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')

def fftshift2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, T, C, H, W)
    return torch.roll(x, shifts=(x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))

def ifftshift2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, T, C, H, W)
    return torch.roll(x, shifts=((x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2), dims=(-2, -1))

def rss(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, T, C, H, W)
    return torch.linalg.vector_norm(x, ord=2, dim=2, keepdim=True)  # Sum over coil dimension

def rss2d(x):
    assert len(x.shape) == 2
    return (x.real ** 2 + x.imag ** 2).sqrt()

# adapted for input including temporal dimension, calculates and averages the SSIM for each frame
def ssimloss(X, Y):
    assert not torch.is_complex(X)
    assert not torch.is_complex(Y)
    
    # Get the temporal dimension size
    num_frames = X.size(1)
    
    # Initialize total SSIM loss
    total_ssim_loss = 0.0
    
    win_size = 7
    k1 = 0.01
    k2 = 0.03
    w = torch.ones(1, 1, win_size, win_size).to(X) / win_size ** 2
    NP = win_size ** 2
    cov_norm = NP / (NP - 1)
    data_range = 1
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    # Iterate over each temporal frame
    for t in range(num_frames):
        # Extract the frame
        Xt = X[:, t, :, :, :]
        Yt = Y[:, t, :, :, :]

        # Compute the means
        ux = F.conv2d(Xt, w)
        uy = F.conv2d(Yt, w)

        # Compute the variances and covariances
        uxx = F.conv2d(Xt * Xt, w)
        uyy = F.conv2d(Yt * Yt, w)
        uxy = F.conv2d(Xt * Yt, w)

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        # Compute the SSIM components
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )

        D = B1 * B2
        S = (A1 * A2) / D

        ssim_loss = 1 - S.mean()
        total_ssim_loss += ssim_loss
    
    # Average the SSIM losses over all frames
    average_ssim_loss = total_ssim_loss / num_frames
    
    return average_ssim_loss

def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=2)  # Concatenate along the coil dimension

def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    _, _, C, _, _ = x.shape
    assert C % 2 == 0
    C = C // 2
    return torch.complex(x[:, :, :C, :, :], x[:, :, C:, :, :])

def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fft2(image * sens_maps)

def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (ifft2(kspace) * sens_maps.conj()).sum(dim=2, keepdim=True)  # Sum over coil dimension

def pad_to_max_size(tensor, max_size):
    """
    Pad the input tensor to the specified max size.
    Args:
        tensor: Input tensor to be padded.
        max_size: The target size to pad the tensor to.
    Returns:
        Padded tensor.
    """
    padding = [0, max_size[i] - tensor.size(i + 2) for i in range(len(max_size))]

    return nn.functional.pad(tensor, padding)

def unpad_from_max_size(tensor, original_size):
    """
    Unpad the input tensor to its original size.
    Args:
        tensor: Input tensor to be unpadded.
        original_size: The original size to unpad the tensor to.
    Returns:
        Unpadded tensor.
    """
    slices = [slice(0, original_size[i]) for i in range(len(original_size))]

    return tensor[..., slices[0], slices[1], slices[2]]

def create_padding_mask(tensor, max_size):
    """
    Create a padding mask for the tensor.
    Valid data is marked with 1, padding with 0.
    """
    mask = torch.ones_like(tensor)
    for i, (dim, max_dim) in enumerate(zip(tensor.shape[2:], max_size)):
        mask[..., :dim] = 1
        mask[..., dim:] = 0
        
    return mask

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

class SpatiotemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatiotemporalAttention, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=(7, 7, 7), stride=1, padding=(3, 3, 3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=2, keepdim=True)
        maxout, _ = torch.max(x, dim=2, keepdim=True)
        out = torch.cat([avgout, maxout], dim=2)
        out = self.sigmoid(self.conv3d(out))
        return out