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
from fastmri import ifft2c, rss, complex_abs

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

        self.norm_net = NormUnet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pools=num_pools,
            drop_prob=drop_prob,
        )
    
    def forward(
        self,
        masked_kspace: torch.Tensor,
        num_low_frequencies: list,
        max_img_size: list
    ) -> torch.Tensor:
        """
        Forward pass with parallel processing over the temporal dimension.
        Args:
            masked_kspace: Input k-space data (with temporal dimension).
            num_low_frequencies: Number of low-frequency components to preserve (list of 1 or 2 values).
        Returns:
            Sensitivity maps and gating parameters.
        """        
        
        # Calculate the ACS mask before padding
        ACS_mask = torch.zeros_like(masked_kspace)

        if len(num_low_frequencies) == 1:
            num_low_freq_w = num_low_frequencies[0]
            center_w = masked_kspace.shape[-2] // 2
            half_num_low_freq_w = num_low_freq_w // 2
            ACS_mask[
                :, :, :,
                :,
                center_w - half_num_low_freq_w: center_w + half_num_low_freq_w,
                :
            ] = 1
        
        elif len(num_low_frequencies) == 2:
            num_low_freq_h, num_low_freq_w = num_low_frequencies
            center_h, center_w = masked_kspace.shape[-3] // 2, masked_kspace.shape[-2] // 2
            half_low_freq_h, half_low_freq_w = num_low_freq_h // 2, num_low_freq_w // 2
            ACS_mask[
                :, :, :, 
                center_h - half_low_freq_h: center_h + half_low_freq_h,
                center_w - half_low_freq_w: center_w + half_low_freq_w,
                :
            ] = 1
        
        masked_kspace_padded, original_size = utils.pad_to_max_size(masked_kspace, max_img_size)
        ACS_mask_padded, _ = utils.pad_to_max_size(ACS_mask, max_img_size)

        # Apply the padded ACS mask to the padded k-space data
        ACS_kspace_padded = ACS_mask_padded * masked_kspace_padded

        # Convert to image space
        ACS_images_padded = ifft2c(ACS_kspace_padded)

        # Reshape to process all temporal slices as a batch
        b, c, t, h, w, comp = ACS_images_padded.shape
        batched_c_t = ACS_images_padded.view(b * c * t, 1, h, w, comp)

        sens, gate = self.norm_net(batched_c_t)

        # Reshape back to original dimensions
        # gate is not complex as we do not want to modulate the real and imaginary parts differently
        gate = gate.reshape(b, c, t, h, w).unsqueeze(-1)
        sens = sens.reshape(b, c, t, h, w, comp)

        # Normalize and apply gating
        sens = sens / (rss(complex_abs(sens), dim=1).unsqueeze(1).unsqueeze(-1) + 1e-12)
        sens = gate * sens
        # NOTE: we will unpad in our model later
        # Unpad the outputs to the original size
        # sensitivity = utils.unpad_from_max_size(sensitivity, original_size)
        # gate = utils.unpad_from_max_size(gate, original_size)

        return sens, gate
    
    
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