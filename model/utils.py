import math

import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F

import fastmri
from fastmri.losses import SSIMLoss

from typing import Tuple

def ssimloss(x, y, max_val):
    # Ensure inputs are the same shape
    assert x.shape == y.shape
    
    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    max_val = max_val.to(device)

    # Reshape the input tensors to combine the batch and temporal dimensions
    B, T, X, Y = x.shape
    x_reshaped = x.view(B * T, 1, X, Y)
    y_reshaped = y.view(B * T, 1, X, Y)
    
    # Instantiate the SSIMLoss class and ensure it's on the correct device
    ssim_loss_fn = SSIMLoss().to(device)
    ssim_loss = ssim_loss_fn(x_reshaped, y_reshaped, max_val)

    return ssim_loss


def normalize_kspace(kspace, epsilon=1e-8):
    # kspace: (B, C, T, H, W, 2)
    # Compute magnitude of complex k-space: sqrt(real^2 + imag^2)
    kspace_magnitude = torch.sqrt(kspace[..., 0]**2 + kspace[..., 1]**2)  # (B, C, T, H, W)

    # Find the maximum magnitude value over the spatial dimensions (H, W)
    max_val = kspace_magnitude.amax(dim=(-2, -1), keepdim=True)  # Reduces over H, W

    # Normalize by the maximum value, adding a small epsilon for numerical stability
    normalized_kspace = kspace_magnitude / (max_val + epsilon)
    return normalized_kspace



def gaussian_kernel_1d(sigma):
    kernel_size = int(2 * math.ceil(sigma * 2) + 1)
    x = torch.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, kernel_size).cuda()
    kernel = 1.0 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_kernel_2d(sigma):
    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])
    kernel = torch.tensordot(y_1, y_2, 0)
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


def gaussian_smooth(img, sigma):
    sigma = max(sigma, 1e-12)
    kernel = gaussian_kernel_2d((sigma, sigma))[None, None, :, :].to(img)
    padding = kernel.shape[-1]//2
    img = torch.nn.functional.conv2d(img, kernel, padding=padding)
    return img


def gaussian_smooth_3d(img, sigma):
    """
    Apply Gaussian smoothing over the spatial (H, W) and temporal (T) dimensions.
    Args:
        img: Input tensor with shape [B, C, T, H, W].
        sigma: A tuple containing the standard deviations for the Gaussian kernel in T, H, and W dimensions.
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
    shape_len = len(x.shape)

    if shape_len == 6:
        b, c, t, h, w, comp = x.shape
        assert comp == 2
        return x.permute(0, 5, 1, 2, 3, 4).reshape(b, comp * c, t, h, w)
    
    elif shape_len == 5:
        b, c, h, w, comp = x.shape
        assert comp == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, comp * c, h, w)


def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    shape_len = len(x.shape)

    if shape_len == 5:
        b, c2, t, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, t, h, w).permute(0, 2, 3, 4, 5, 1).contiguous()
    
    elif shape_len == 4:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return fastmri.complex_mul(
        fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
    ).sum(dim=1, keepdim=True)


def chans_to_batch_dim(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Flattens the batch and coil dimensions into a single dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (b, c, h, w, comp).

    Returns:
        Tuple[torch.Tensor, int]: 
            - Flattened tensor of shape (b * c, 1, h, w, comp).
            - Original batch size `b`.
    """
    b, c, h, w, comp = x.shape
    return x.view(b * c, 1, h, w, comp), b


def batch_chans_to_chan_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Restores the original batch and coil dimensions from a flattened dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (b * c, 1, h, w, comp).
        batch_size (int): Original batch size.

    Returns:
        torch.Tensor: Reshaped tensor of shape (b, c, h, w, comp).
    """
    bc, _, h, w, comp = x.shape
    c = bc // batch_size
    return x.view(batch_size, c, h, w, comp)


def pad_to_max_size(tensor, max_size):
    """
    Pad the input tensor to the specified max size, with even padding on both sides
    of the spatial dimensions to keep the subject centered.

    Args:
        tensor: Input tensor of shape image--> (B, C, T, H, W, comp) or mask--> (B, T, H, W, comp).
        max_size: The target size to pad the tensor to, in the form (T, H, W).
    Returns:
        Padded tensor and the original shape of the tensor.
    """
    # Record the original size
    original_size = list(tensor.size())

    # Calculate padding for temporal, height, and width dimensions
    pad_t = max_size[0] - tensor.size(-4)
    pad_h = max_size[1] - tensor.size(-3)
    pad_w = max_size[2] - tensor.size(-2)

    # Even padding for temporal dimension
    pad_t_left = pad_t // 2
    pad_t_right = pad_t - pad_t_left

    # Even padding for height and width dimensions, handling odd padding
    pad_h_left = pad_h // 2
    pad_h_right = pad_h - pad_h_left
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left

    # PyTorch padding order: [pad_t_left, pad_t_right, pad_h_left, pad_h_right, pad_w_left, pad_w_right]
    padding = [
        0, 0,                      # Comp dimension
        pad_w_left, pad_w_right,   # W dimension
        pad_h_left, pad_h_right,   # H dimension
        pad_t_left, pad_t_right    # T dimension
    ]

    padded_tensor = nn.functional.pad(tensor, padding)

    return padded_tensor, original_size



def unpad_from_max_size(tensor, original_size):
    """
    Unpad the input tensor to its original size, ensuring that the odd paddings
    from the `pad_to_max_size` function are correctly handled.

    Args:
        tensor: Input tensor to be unpadded.
        original_size: The original size to unpad the tensor to.
    Returns:
        Unpadded tensor.
    """
    # Extract the original dimensions
    original_t, original_h, original_w = original_size[-4], original_size[-3], original_size[-2]

    # Calculate the starting indices for unpadding
    start_t = (tensor.size(-4) - original_t) // 2
    start_h = (tensor.size(-3) - original_h) // 2
    start_w = (tensor.size(-2) - original_w) // 2

    # Handle the odd paddings by slicing exactly up to the original size
    return tensor[
        :, :, 
        start_t : start_t + original_t, 
        start_h : start_h + original_h, 
        start_w : start_w + original_w, :
    ]



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



def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2),
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1], indices[3][0]:indices[3][1]]

def crop_submission(a, attrs):
    # Determine modality type
    modality_view = attrs.get('modality_view', '').lower()
    if 'cine' in modality_view:
        modality_type = 'cine'
    elif 't1map' in modality_view:
        modality_type = 'T1map'
    elif 't2map' in modality_view:
        modality_type = 'T2map'
    else:
        modality_type = 'other'

    sx, sy, sz, st = a.shape
    
    # Central slices logic
    if sz >= 3:
        start_slice = (sz // 2) - 1
        end_slice = start_slice + 2
        a = a[:, :, start_slice:end_slice, :]
    
    # Time frames logic
    if modality_type == 'cine':
        # Use first 3 time frames
        a = a[..., :3]
    elif modality_type in ['T1map','T2map']:
        # Use all time frames
        pass  # a stays unchanged for time dimension
    else:
        # For black-blood or other data:
        a = a[..., :1]
    
    # Spatial cropping
    cropped = _crop(a, (int(round(sx/3)), int(round(sy/2)), a.shape[2], a.shape[3]))
    return cropped

