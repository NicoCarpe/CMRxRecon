from __future__ import division
import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fastmri
from fastmri import SSIMLoss
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio




def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return ((np.linalg.norm((gt - pred).cpu()) ** 2) / ((np.linalg.norm(gt.cpu()) ** 2) + 1e-8))    # ensure numerical stability


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    return peak_signal_noise_ratio(gt_np, pred_np, data_range=maxval)


def ssimloss(gt, pred, max_val):
    # Ensure inputs are not complex
    assert not torch.is_complex(gt)
    assert not torch.is_complex(pred)
    
    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reshape the input tensors to combine the batch and temporal dimensions
    B, T, X, Y = gt.shape
    gt_reshaped = gt.view(B * T, 1, X, Y).to(device)
    pred_reshaped = pred.view(B * T, 1, X, Y).to(device)
    max_val = max_val.to(device)
    
    # Instantiate the SSIMLoss class and ensure it's on the correct device
    ssim_loss_fn = SSIMLoss().to(device)
    ssim_loss = ssim_loss_fn(gt_reshaped, pred_reshaped, max_val)

    return ssim_loss


def combined_loss(rec_img, target_img, sens_maps, masked_kspace, mask, max_val, alpha=0.5, beta=0.5, lambda_SM=0.1):
    # SSIM and L1 Loss for Image Reconstruction
    loss_ssim = ssimloss(rec_img, target_img, max_val)
    loss_l1_image = F.l1_loss(rec_img, target_img)
    
    # Weighted combination of SSIM and L1 for image loss
    loss_image = beta * loss_ssim + (1 - beta) * loss_l1_image
    
    # L1 Loss for Sensitivity Map Consistency
    # NOTE: mask and sens_maps both have the c and comp dims, need to match with the image
    loss_sensitivity_consistency = F.l1_loss(mask * sens_expand(rec_img.unsqueeze(1).unsqueeze(-1), sens_maps), masked_kspace)
    
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
    return fastmri.fft2c(image * sens_maps)


def sens_reduce(kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return (fastmri.ifft2c(kspace) * sens_maps.conj()).sum(dim=1, keepdim=True)  # Sum over coil dimension


def pad_to_max_size2D(tensor, max_size):
    """
    Pad the input tensor to the specified max size.
    Args:
        tensor: Input tensor of shape (B, C, X, Y, 2) or (B, X, Y, 2) to be padded.
        max_size: The target size to pad the tensor to, in the form (X, Y).
    Returns:
        Padded tensor and the original shape of the tensor.
    """
    
    # Record the original size
    original_size = list(tensor.size())

    # Pad the tensor
    # PyTorch's padding expects the order [pad_y_left, pad_y_right, pad_x_left, pad_x_right, pad_t_left, pad_t_right]
    # No need to pad the last complex dimension (2).
    padding = [
        0, 0,  # No padding for the complex dimension
        0, max_size[1] - tensor.size(-1),  # y dimension
        0, max_size[0] - tensor.size(-2),  # x dimension
    ]

    padded_tensor = nn.functional.pad(tensor, padding)

    return padded_tensor, original_size


def unpad_from_max_size2D(tensor, original_size):
    """
    Unpad the input tensor to its original size.
    Args:
        tensor: Input tensor to be unpadded.
        original_size: The original size to unpad the tensor to.
    Returns:
        Unpadded tensor.
    """
    return tensor[:, :, :original_size[2], :original_size[3], :]


def pad_to_max_size(tensor, max_size):
    """
    Pad the input tensor to the specified max size.
    Args:
        tensor: Input tensor of shape (B, C, T, X, Y, 2) or (B, T, X, Y, 2) to be padded.
        max_size: The target size to pad the tensor to, in the form (T, X, Y).
    Returns:
        Padded tensor and the original shape of the tensor.
    """
    
    # Record the original size
    original_size = list(tensor.size())

    # Pad the tensor
    # PyTorch's padding expects the order [pad_y_left, pad_y_right, pad_x_left, pad_x_right, pad_t_left, pad_t_right]
    # No need to pad the last complex dimension (2).
    padding = [
        0, 0,  # No padding for the complex dimension
        0, max_size[2] - tensor.size(-2),  # y dimension
        0, max_size[1] - tensor.size(-3),  # x dimension
        0, max_size[0] - tensor.size(-4)   # t dimension
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
    return tensor[:, :, :original_size[2], :original_size[3], :original_size[4], :]
