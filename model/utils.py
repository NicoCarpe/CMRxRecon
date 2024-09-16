"""
@author: sunkg

Extended by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""

import math
import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F
# from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from fastmri.losses import SSIMLoss


def fft2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, X, Y)
    return torch.fft.fft2(x, dim=(-2, -1), norm='ortho')


def ifft2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, X, Y)
    return torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')


def fftshift2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, X, Y)
    return torch.roll(x, shifts=(x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))


def ifftshift2(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, X, Y)
    return torch.roll(x, shifts=((x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2), dims=(-2, -1))


def rss(x):
    assert len(x.shape) == 5  # Ensure the input is 5D (B, C, T, X, Y)
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)  # Sum over coil dimension


def rss2d(x):
    assert len(x.shape) == 2
    return (x.real ** 2 + x.imag ** 2).sqrt()


def ssimloss(x, y, max_val):
    # Ensure inputs are not complex
    assert not torch.is_complex(x)
    assert not torch.is_complex(y)
    
    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    max_val = max_val.to(device)

    # Reshape the input tensors to combine the batch and temporal dimensions
    B, C, T, X, Y = x.shape
    x_reshaped = x.view(B * T, C, X, Y)
    y_reshaped = y.view(B * T, C, X, Y)
    
    # Instantiate the SSIMLoss class and ensure it's on the correct device
    ssim_loss_fn = SSIMLoss().to(device)
    ssim_loss = ssim_loss_fn(x_reshaped, y_reshaped, max_val)

    return ssim_loss


def psnr(mse_loss, max_val):
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse_loss)
    return psnr


def combined_loss(rec_img, target_img, sens_maps, masked_kspace, mask, max_val, alpha=0.5, beta=0.5, lambda_SM=0.1):
    # SSIM and L1 Loss for Image Reconstruction
    loss_ssim = ssimloss(rec_img, target_img, max_val)
    loss_l1_image = F.l1_loss(rec_img, target_img)
    
    # Weighted combination of SSIM and L1 for image loss
    loss_image = beta * loss_ssim + (1 - beta) * loss_l1_image
    
    # L1 Loss for Sensitivity Map Consistency
    loss_sensitivity_consistency = F.l1_loss(mask * sens_expand(rec_img, sens_maps), masked_kspace)
    
    # Final combined loss: Image loss with alpha scaling + Sensitivity Map loss with lambda scaling
    combined_loss = alpha * loss_image + lambda_SM * loss_sensitivity_consistency
    
    return combined_loss


def gaussian_kernel_1d(sigma):
    kernel_size = int(2 * math.ceil(sigma * 2) + 1)
    x = torch.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, kernel_size).cuda()
    kernel = 1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_kernel_3d(sigma):
    # Create 1D Gaussian kernels for each dimension
    kernel_t = gaussian_kernel_1d(sigma[0])
    kernel_x = gaussian_kernel_1d(sigma[1])
    kernel_y = gaussian_kernel_1d(sigma[2])
    
    # Create a 3D Gaussian kernel by taking the outer product
    kernel = torch.einsum('i,j,k->ijk', kernel_t, kernel_x, kernel_y)
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_smooth_3d(img, sigma):
    """
    Apply Gaussian smoothing over the spatial (X, Y) and temporal (T) dimensions.
    Args:
        img: Input tensor with shape [B, C, T, X, Y].
        sigma: A tuple containing the standard deviations for the Gaussian kernel in T, X, and Y dimensions.
    Returns:
        smoothed_img: Smoothed tensor with the same shape as the input.
    """
    sigma = [max(s, 1e-12) for s in sigma]
    kernel = gaussian_kernel_3d(sigma)[None, None, :, :, :].to(img)
    padding = [(k - 1) // 2 for k in kernel.shape[-3:]]
    
    # Apply the 3D Gaussian smoothing
    smoothed_img = torch.nn.functional.conv3d(img, kernel, padding=padding)
    
    return smoothed_img


def TV_loss(img, weight):
    bs_img, c_img, t_img, x_img, y_img = img.size()
    tv_t = torch.pow(img[:, :, 1:, :, :] - img[:, :, :-1, :, :], 2).sum()
    tv_x = torch.pow(img[:, :, :, 1:, :] - img[:, :, :, :-1, :], 2).sum()
    tv_y = torch.pow(img[:, :, :, :, 1:] - img[:, :, :, :, :-1], 2).sum()
    return (weight * (tv_t + tv_x + tv_y)) / (bs_img * c_img * t_img * x_img * y_img)


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    assert torch.is_complex(x)
    return torch.cat([x.real, x.imag], dim=1)  # Concatenate along the channel dimension


def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    assert not torch.is_complex(x)
    _, C, _, _, _ = x.shape
    assert C % 2 == 0
    C = C // 2
    return torch.complex(x[:, :C, :, :, :], x[:, C:, :, :, :])


def sens_expand(image: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fft2(image * sens_maps)


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (ifft2(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)  # Sum over coil dimension


def pad_to_max_size(tensor, max_size):
    """
    Pad the input tensor to the specified max size.
    Args:
        tensor: Input tensor of shape (B, C, T, X, Y) or (B, T, X, Y) to be padded.
        max_size: The target size to pad the tensor to, in the form (T, X, Y).
    Returns:
        Padded tensor and the original shape of the tensor.
    """
    
    # Record the original size
    original_size = list(tensor.size())

    # Pad the tensor
    # PyTorch's padding expects order 
    # [pad_t_left, pad_t_right, pad_x_left, pad_x_right, pad_y_left, pad_y_right]
    padding = [
        0, max_size[2] - tensor.size(-1),  # y dimension
        0, max_size[1] - tensor.size(-2),  # x dimension
        0, max_size[0] - tensor.size(-3)   # t dimension
    ]

    padded_tensor = nn.functional.pad(tensor, padding)

    return padded_tensor, original_size


def unpad_from_max_size(tensor, original_size):
    """
    Unpad the input tensor to its original size.
    Args:
        tensor: Input tensor to be unpadded.
        original_size: The original size to unpad the tensor to.
    Returns:
        Unpadded tensor.
    """
    return tensor[:, :, :original_size[2], :original_size[3], :original_size[4]]


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