import math

import torch.nn as nn
import torch
import torch.fft
import torch.nn.functional as F

import fastmri
from fastmri.losses import SSIMLoss


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


def combined_loss(target_img, rec_img, max_val, loss_sc, alpha=0.5, beta=0.5, lambda_SM=0.1):
    # SSIM and L1 Loss for Image Reconstruction
    loss_ssim = ssimloss(rec_img, target_img, max_val)
    loss_l1_image = F.l1_loss(rec_img, target_img)
    
    # Weighted combination of SSIM and L1 for image loss
    loss_image = beta * loss_ssim + (1 - beta) * loss_l1_image
    
    # Final combined loss: Image loss with alpha scaling + Sensitivity Map loss with lambda scaling
    combined_loss = alpha * loss_image + lambda_SM * loss_sc
    
    return combined_loss


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
        b, c, t, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 5, 1, 2, 3, 4).reshape(b, 2 * c, t, h, w)
    
    elif shape_len == 5:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


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


def pad_to_max_size(tensor, max_size):
    """
    Pad the input tensor to the specified max size.
    Args:
        tensor: Input tensor of shape: image--> (B, C, T, H, W, comp) or mask--> (B, T, H, W, comp) to be padded.
        max_size: The target size to pad the tensor to, in the form (T, H, W).
    Returns:
        Padded tensor and the original shape of the tensor.
    """
    # Record the original size
    original_size = list(tensor.size())

    # Pad the tensor
    # PyTorch's padding expects order 
    # [pad_t_left, pad_t_right, pad_h_left, pad_h_right, pad_w_left, pad_w_right]
    padding = [
        0, 0,                              # comp dim needs to be accounted for
        0, max_size[2] - tensor.size(-2),  # w dimension
        0, max_size[1] - tensor.size(-3),  # h dimension
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