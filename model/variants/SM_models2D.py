#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:21:26 2022

@author: sunkg
""" 

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from model import utils
from data_utils.transforms import batched_mask_center
from fastmri import ifft2c, rss, complex_abs, rss_complex

class SMEB(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. 

    Note SMEB is designed for complex input/output only.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 32,
        num_pools: int = 4,
        drop_prob: float = 0.0,
        mask_center: bool = True 
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center

        self.norm_unet = NormUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pools=num_pools,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        
        return x.view(b * c, 1, h, w, comp), b


    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)
    
    
    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)


    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = mask.shape[0]
        pad = torch.zeros((batch_size, 2), dtype=torch.long, device=mask.device)  # Ensure pad has shape [batch_size, 2]

        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            ).view(batch_size, 1)  # Shape it as [batch_size, 1] for compatibility
            
            pad[:, 1] = ((mask.shape[-2] - num_low_frequencies_tensor + 1) // 2).squeeze(1)

        else:
            num_low_frequencies_tensor = torch.ones(
                (batch_size, 2), dtype=torch.long, device=mask.device
            )

            if len(num_low_frequencies) == 1:
                # Only height is specified
                num_low_frequencies_tensor[:, 0] = num_low_frequencies[0]  # Height dimension
                pad[:, 0] = ((mask.shape[-3] - num_low_frequencies_tensor[:, 0] + 1) // 2).long()

                # For width, we don't change anything, so full width is used
                num_low_frequencies_tensor[:, 1] = mask.shape[-2]  # Full width
                pad[:, 1] = 0  # No padding for width

            elif len(num_low_frequencies) == 2:
                num_low_frequencies_tensor[:, 0] = num_low_frequencies[0]  # h dimension
                num_low_frequencies_tensor[:, 1] = num_low_frequencies[1]  # w dimension
                pad[:, 0] = ((mask.shape[-3] - num_low_frequencies_tensor[:, 0] + 1) // 2).long()
                pad[:, 1] = ((mask.shape[-2] - num_low_frequencies_tensor[:, 1] + 1) // 2).long()

        return pad, num_low_frequencies_tensor
    

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: list,
        max_img_size: list
    ) -> torch.Tensor:
        """
        Forward pass with parallel processing over the temporal dimension using SMEB logic.
        Args:
            masked_kspace: Input k-space data (with temporal dimension).
            mask: K-space mask.
            num_low_frequencies: Number of low-frequency components to preserve.
            max_img_size: Maximum image size for padding.
        Returns:
            Sensitivity maps and gating parameters.
        """
        b, c, t, h, w, comp = masked_kspace.shape

        # Mask the center frequencies
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask[:, :, 0, :, :, :], num_low_frequencies)
            masked_kspace = masked_kspace.clone()
            for time_idx in range(t):
                modified_kspace = batched_mask_center(
                    masked_kspace[:, :, time_idx, :, :, :], pad, pad + num_low_freqs
                )
                masked_kspace[:, :, time_idx, :, :, :] = modified_kspace

        # Pad without modifying the original tensor
        masked_kspace, _ = utils.pad_to_max_size(masked_kspace, max_img_size)
        b, c, t, h, w, comp = masked_kspace.shape

        # Convert k-space to image domain
        images = ifft2c(masked_kspace)

        # Reshape to process all temporal slices as a batch
        batched_images, batch_size = self.chans_to_batch_dim(images.view(b, c * t, h, w, comp))

        # Pass through the U-Net
        unet_output, gates = self.norm_unet(batched_images)

        # Reshape the U-Net output back to the original dimensions
        unet_output = self.batch_chans_to_chan_dim(unet_output, batch_size)
        gates = gates.view(b, c, t, h, w, 1)

        # Normalize sensitivities and apply gating
        rss_norm = rss(complex_abs(unet_output), dim=1).unsqueeze(1).unsqueeze(-1)
        sens = unet_output / (rss_norm + 1e-12)
        sens = sens * gates

        return sens, gates

    
    
class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int,
        num_pools: int,
        drop_prob: float,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pools=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        #### gated ####
        x, gate = self.unet(x)
        gate = self.unpad(gate, *pad_sizes)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x, gate
   

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int,
        num_pools: int,
        drop_prob: float,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob

        # Initialize the downsampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans

        for _ in range(num_pools - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # Initialize the upsampling path
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        for _ in range(num_pools - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.conv_sl = ConvBlock(ch*2, ch, drop_prob)
        self.norm_conv = nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1)

        #### gated conv ####    
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.gate_conv = nn.Conv2d(ch, 1, kernel_size=1, stride=1)

        #### learnable Gaussian std ####
        self.stds = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_pools)])


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        cnt = 0

        # apply up-sampling layers
        for transpose_conv, up_conv in zip(self.up_transpose_conv, self.up_conv):
 
            #### learnable Gaussian std ####
            output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-2:]), sigma = torch.clamp(self.stds[cnt], min=1e-9, max=7))
            output = output_.view(output.shape)
            
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom

            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = up_conv(output)

            cnt = cnt + 1

        output = self.conv_sl(output)

        #### learnable Gaussian std ####
        output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-2:]), sigma = torch.clamp(self.stds[cnt], min=1e-9, max=7))
        output = output_.view(output.shape)

        # TODO: This is a little ugly, need to adjust the hardcoding
        output = F.interpolate(output, scale_factor=2**(self.num_pools-3))  

        #### normal ####
        norm_conv = self.norm_conv(output)

        #### gated ####
        gate_conv = torch.sigmoid(self.scale * (self.gate_conv(output) + self.shift))
       
        return norm_conv, gate_conv

        
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(
        self, 
        in_chans: int, 
        out_chans: int, 
        drop_prob: float
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(B, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, H, W)`.
        """
        return self.layers(image)

    
class ConvBlockSM(nn.Module):
    """
    light-weight cascade of two convolutional layers with kernel size of 3 × 3 
    based on skip connection followed by spatial attention
    """
   
    def __init__(
        self, 
        in_chans: int = 2, 
        out_chans: int = 2, 
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        
        self.SA = SpatialAttention()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, H, W)`.
        """
        output = self.layers(image)

        #### Spatial Attention ###    
        output = self.SA(output)*output  

        return output
    
    
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(
        self, 
        in_chans: int, 
        out_chans: int
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(B, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
    

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