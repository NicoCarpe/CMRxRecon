#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:21:26 2022

@author: sunkg

Jul 23, 2024
Extended by Nicolas Carpenter <ngcarpen@ualberta.ca>
""" 

import torch.nn as nn
import torch
import utils
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# Transpose blocks were not used in this code originally, so make sure the change is necessary
from CNN import ConvBlock3D, TransposeConvBlock3D

#########################################################################################################
# Sensitivity Model
#########################################################################################################

class SMEB3D(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.

    Note SensitivityModel is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        sens_steps: int,
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

        self.norm_net = NormNet3D(
            chans,
            sens_steps
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        num_low_frequencies: list,
        max_img_size: tuple
    ) -> torch.Tensor:
        # Calculate the ACS mask before padding
        if len(num_low_frequencies) == 1:
            num_low_freq_y = num_low_frequencies[0]
            center_y = masked_kspace.shape[-1] // 2
            half_num_low_freq_y = num_low_freq_y // 2
            ACS_mask = torch.zeros_like(masked_kspace)
            ACS_mask[..., :, center_y - half_num_low_freq_y: center_y + half_num_low_freq_y] = 1
        
        elif len(num_low_frequencies) == 2:
            num_low_freq_x, num_low_freq_y = num_low_frequencies
            center_x, center_y = masked_kspace.shape[-2] // 2, masked_kspace.shape[-1] // 2
            half_low_freq_x, half_low_freq_y = num_low_freq_x // 2, num_low_freq_y // 2
            ACS_mask = torch.zeros_like(masked_kspace)
            ACS_mask[
                ..., center_x - half_low_freq_x: center_x + half_low_freq_x,
                center_y - half_low_freq_y: center_y + half_low_freq_y
            ] = 1

        # Pad the k-space data and the ACS mask
        original_size = masked_kspace.shape[2:]  # Exclude batch and coil dimensions
        masked_kspace_padded = utils.pad_to_max_size(masked_kspace, max_img_size)
        ACS_mask_padded = utils.pad_to_max_size(ACS_mask, max_img_size)

        # Apply the padded ACS mask to the padded k-space data
        ACS_kspace_padded = ACS_mask_padded * masked_kspace_padded

        # Convert to image space
        ACS_images_padded = utils.ifft2(ACS_kspace_padded)

        # Estimate sensitivities independently
        B, T, C, H, W = ACS_images_padded.shape
        batched_channels = ACS_images_padded.reshape(B * T, 1, C, H, W)

        sensitivity, gate = self.norm_net(batched_channels)
        gate = gate.reshape(B, T, C, H, W)
        sensitivity = sensitivity.reshape(B, T, C, H, W)
        sensitivity = sensitivity / (utils.rss(sensitivity) + 1e-12)
        sensitivity = gate * sensitivity

        # Unpad the outputs to the original size
        sensitivity = utils.unpad_from_max_size(sensitivity, original_size)
        gate = utils.unpad_from_max_size(gate, original_size)

        return sensitivity, gate
    
#########################################################################################################
# 3D UNet
#########################################################################################################
class Unet3D(nn.Module):
    """
    PyTorch implementation of a 3D U-Net model for 2D+t data.

    #TODO update for 3D UNet reference

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        chans: int = 32,
        num_pool_layers: int = 4,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = 2
        self.out_chans = 2
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock3D(self.in_chans, chans)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2))
            ch *= 2
        self.conv = ConvBlock3D(ch, ch * 2)

        self.up_conv = nn.ModuleList()
        self.upsampling_conv = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            self.upsampling_conv.append(TransposeConvBlock3D(ch * 2, ch))
            self.up_conv.append(ConvBlock3D(ch * 2, ch))
            ch //= 2

        self.upsampling_conv.append(TransposeConvBlock3D(ch * 2, ch))

        self.conv_sl = ConvBlock3D(ch * 2, ch)
        self.norm_conv = nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)

        #### gated conv ####    
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.gate_conv = nn.Conv3d(ch, 1, kernel_size=1, stride=1)

        #### learnable Gaussian std ####
        self.stds = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_pool_layers)])


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, T, C, H, W)`.

        Returns:
            Output tensor of shape `(B, T, out_chans, H, W)`.
        """
        assert not torch.is_complex(image)
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)

        output = self.conv(output)
        cnt = 0
        # apply up-sampling layers
        for up_conv, conv in zip(self.upsampling_conv, self.up_conv):

            #### learnable Gaussian std ####
            output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-3:]), sigma=torch.clamp(self.stds[cnt], min=1e-9, max=7))
            output = output_.view(output.shape)

            downsample_layer = stack.pop()
            output = up_conv(output)
            output = F.interpolate(output, scale_factor=(2, 2, 1))

            # reflect pad on the right/bottom if needed to handle odd input dimensions
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
            cnt = cnt + 1

        output = self.conv_sl(output)
        #### learnable Gaussian std ####
        output_ = utils.gaussian_smooth(output.view(-1, 1, *output.shape[-3:]), sigma=torch.clamp(self.stds[cnt], min=1e-9, max=7))
        output = output_.view(output.shape)
        output = F.interpolate(output, scale_factor=(2**(self.num_pool_layers-3), 2**(self.num_pool_layers-3), 1))
        norm_conv = self.norm_conv(output)

        #### gated ####
        gate_conv = torch.sigmoid(self.scale * (self.gate_conv(output) + self.shift))

        return norm_conv, gate_conv

#########################################################################################################
# 3D NormNet
#########################################################################################################

class NormNet3D(nn.Module):
    """
    Normalized Net model: in Unet or ResNet

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
        self,
        chans: int,
        num_steps: int
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
        """
        super().__init__()
        
        self.NormNet3D = Unet3D(
            chans=chans,
            num_pool_layers=num_steps,
        )


    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(x)
        return torch.cat([x.real, x.imag], dim=1)


    def chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(x)
        _, c, _, _, _ = x.shape
        assert c % 2 == 0
        c = c // 2
        return torch.complex(x[:, :c], x[:, c:])


    def norm(self, x: torch.Tensor):
        # group norm
        B, T, C, H, W = x.shape
        assert C % 2 == 0
        x = x.view(B, T, 2, C // 2 * H * W)

        mean = x.mean(dim=3).view(B, T, 2, 1)
        std = x.std(dim=3).view(B, T, 2, 1)

        x = (x - mean) / (std + 1e-12)

        return x.view(B, T, C, H, W), mean, std


    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        B, T, C, H, W = x.shape
        assert C % 2 == 0
        x = x.contiguous().view(B, T, 2, C // 2 * H * W)
        x = x * std + mean
        return x.view(B, T, C, H, W)


    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], List[int], int, int, int]]:
        _, _, t, h, w = x.shape
        t_mult = ((t - 1) | 15) + 1
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        t_pad = [math.floor((t_mult - t) / 2), math.ceil((t_mult - t) / 2)]
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, t_pad + w_pad + h_pad)

        return x, (t_pad, h_pad, w_pad, t_mult, h_mult, w_mult)


    def unpad(
        self,
        x: torch.Tensor,
        t_pad: List[int],
        h_pad: List[int],
        w_pad: List[int],
        t_mult: int,
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., t_pad[0]: t_mult - t_pad[1], h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]


    def forward(
        self, 
        x: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(x.shape) == 5
        assert torch.is_complex(x)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
            
        #### gated ####
        x, gate = self.NormNet3D(x)
        gate = self.unpad(gate, *pad_sizes)
        
        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_dim_to_complex(x)

        return x, gate
