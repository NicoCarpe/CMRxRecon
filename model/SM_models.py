#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 07:21:26 2022

@author: sunkg

2024
Extended to 3D by Nicolas Carpenter <ngcarpen@ualberta.ca>
""" 

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional
from model import utils
from data_utils.transforms import batched_mask_center
from fastmri import ifft2c, rss, complex_abs

#########################################################################################################
# Sensitivity Model
#########################################################################################################

class SMEB3D(nn.Module):
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

        # Replace NormUnet with a 3D U-Net
        self.norm_net = NormNet3D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pools=num_pools,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, t, h, w, comp = x.shape
        # Batch over the channel dimension
        return x.view(b * c, 1, t, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, t, h, w, comp = x.shape
        c = bc // batch_size
        # Restore the channel dimension
        return x.view(batch_size, c, t, h, w, comp)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = mask.shape[0]
        pad = torch.zeros((batch_size, 2), dtype=torch.long, device=mask.device)

        if num_low_frequencies is None or num_low_frequencies == 0:
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            ).view(batch_size, 1)
            pad[:, 1] = ((mask.shape[-2] - num_low_frequencies_tensor + 1) // 2).squeeze(1)

        else:
            num_low_frequencies_tensor = torch.ones(
                (batch_size, 2), dtype=torch.long, device=mask.device
            )
            if len(num_low_frequencies) == 1:
                num_low_frequencies_tensor[:, 0] = num_low_frequencies[0]
                pad[:, 0] = ((mask.shape[-3] - num_low_frequencies_tensor[:, 0] + 1) // 2).long()
                num_low_frequencies_tensor[:, 1] = mask.shape[-2]
                pad[:, 1] = 0
            elif len(num_low_frequencies) == 2:
                num_low_frequencies_tensor[:, 0] = num_low_frequencies[0]
                num_low_frequencies_tensor[:, 1] = num_low_frequencies[1]
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
        b, c, t, h, w, comp = masked_kspace.shape

        # Mask the center frequencies
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask[:, :, 0, :, :, :], num_low_frequencies)
            masked_kspace = masked_kspace.clone()
            modified_kspace = batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
            masked_kspace = modified_kspace

        # Pad without modifying the original tensor
        masked_kspace, _ = utils.pad_to_max_size(masked_kspace, max_img_size)
        b, c, t, h, w, comp = masked_kspace.shape
        
        # Convert k-space to image domain
        images = ifft2c(masked_kspace)

        # Reshape to batch over the channel dimension
        batched_images, batch_size = self.chans_to_batch_dim(images)

        # Pass through the 3D U-Net
        unet_output, gates = self.norm_net(batched_images)

        # Restore the channel dimension
        unet_output = self.batch_chans_to_chan_dim(unet_output, batch_size)
        gates = gates.view(b, c, t, h, w, 1)

        # Normalize sensitivities and apply gating
        rss_norm = rss(complex_abs(unet_output), dim=1).unsqueeze(1).unsqueeze(-1)
        sens = unet_output / (rss_norm + 1e-12)
        sens = sens * gates

        return sens, gates

    

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
        
        self.unet_3d = UNet3D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pools=num_pools,
            drop_prob=drop_prob,
        )


    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, t, h, w = x.shape
        x = x.view(b, 2, c // 2 * t * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1)

        x = x.view(b, c, t, h, w)

        return (x - mean) / std, mean, std


    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], List[int], int, int, int]]:
        _, _, t, h, w = x.shape

        # Calculate the multiples to align to the nearest multiple of 16
        t_mult = ((t - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_mult = ((w - 1) | 15) + 1

        # Calculate padding required for each dimension
        t_pad = [math.floor((t_mult - t) / 2), math.ceil((t_mult - t) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]

        # Apply padding symmetrically
        x = F.pad(x, w_pad + h_pad + t_pad, mode="constant", value=0)

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
    ) -> torch.Tensor:
        
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # Convert to real-valued tensor and normalize
        x = utils.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        # Pass through UNet3D (NormNet3D)
        x, gate = self.unet_3d(x)
        gate = self.unpad(gate, *pad_sizes)
        
        # Unnormalize and convert back to complex
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = utils.chan_complex_to_last_dim(x)

        return x, gate
    

#########################################################################################################
# 3D UNet 
#########################################################################################################

class UNet3D(nn.Module):
    """
    PyTorch implementation of a 3D U-Net model.

    Çiçek, , Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3d u-net:
    Learning dense volumetric segmentation from sparse annotation. Proceedings of
    MICCAI pp. 424–432 (2016). https://doi.org/10.1007/978-3-319-46723-8_49
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int,
        num_pools: int,
        drop_prob: float,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pools = num_pools
        self.drop_prob = drop_prob

        # Initialize the downsampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_chans, chans, drop_prob)])
        ch = chans

        for _ in range(self.num_pools - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock3D(ch, ch * 2, drop_prob)

        # Initialize the upsampling path
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        for _ in range(self.num_pools - 1):
            self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch))
            self.up_conv.append(ConvBlock3D(ch * 2, ch, drop_prob))
            ch //= 2


        self.conv_sl = ConvBlock3D(ch*2, ch, drop_prob)
        self.norm_conv = nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)

        # Gated Convolution
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.gate_conv = nn.Conv3d(ch, 1, kernel_size=1, stride=1)

        # Learnable Gaussian std
        self.stds = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_pools)])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(B, in_chans, T, H, W)`.
        """
        stack = []
        output = image

        # Downsampling path
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        # Bottleneck
        output = self.conv(output)
        cnt = 0
            
        # Upsampling path
        for transpose_conv, up_conv in zip(self.up_transpose_conv, self.up_conv):

            #### learnable Gaussian std ####
            sigma_value = torch.clamp(self.stds[cnt], min=1e-9, max=7)
            sigma = (0, sigma_value.item(), sigma_value.item())  # No smoothing on the temporal dimension
            output_ = utils.gaussian_smooth_3d(output.view(-1, 1, *output.shape[-3:]), sigma=sigma)
            output = output_.view(output.shape)

            downsample_layer = stack.pop()
            output = transpose_conv(output)
            
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding W dimension
            if output.shape[-2] != downsample_layer.shape[-2]:  
                padding[3] = 1  # padding H dimension
            if output.shape[-3] != downsample_layer.shape[-3]: 
                padding[5] = 1  # padding T dimension

            if any(padding):
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = up_conv(output)

            cnt += 1

        output = self.conv_sl(output)

        #### learnable Gaussian std ####
        sigma_value = torch.clamp(self.stds[cnt], min=1e-9, max=7)
        sigma = (0, sigma_value.item(), sigma_value.item())  # No smoothing on the temporal dimension
        output_ = utils.gaussian_smooth_3d(output.view(-1, 1, *output.shape[-3:]), sigma=sigma)
        output = output_.view(output.shape)

        # TODO: This is a little ugly, need to adjust the hardcoding
        output = F.interpolate(output, scale_factor=2**(self.num_pools-3), mode='trilinear', align_corners=False)

        #### normal ####
        norm_conv = self.norm_conv(output)

        #### gated ####
        gate_conv = torch.sigmoid(self.scale * (self.gate_conv(output) + self.shift))

        return norm_conv, gate_conv



#########################################################################################################
# 3D Convolution 
#########################################################################################################

class ConvBlock3D(nn.Module):
    """
    A Convolutional Block that consists of a convolution layer followed by
    instance normalization and LeakyReLU activation.
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
            nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, H, W)`.
        """
        return self.layers(image)


#########################################################################################################
# 3D Transpose Convolution
#########################################################################################################
    
class TransposeConvBlock3D(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layer followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=2, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, T*2, H*2, W*2)`.
        """
        return self.layers(image)
    

#########################################################################################################
# 3D SM Convolution w/ Spatiotemporal Attention
#########################################################################################################

class ConvBlockSM3D(nn.Module):
    """
    A Convolutional Block that consists of convolution layers each followed by
    instance normalization, LeakyReLU activation, and spatiotemporal attention.
    """

    def __init__(
        self, 
        in_chans: int = 2, 
        out_chans: int = 2, 
        drop_prob: float = 0.0
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.layers = (nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
        ))

        #### Spatiotemporal Attention ####
        self.STA = SpatiotemporalAttention(in_channels=in_chans)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, C, T, H, W)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, H, W)`.
        """
        output = self.layers(image)

        #### Spatiotemporal Attention ###
        output = self.STA(output) * output

        return output


#########################################################################################################
# Attention Mechanisms
#########################################################################################################

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        avgout = torch.mean(image, dim=1, keepdim=True)
        maxout, _ = torch.max(image, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# class SpatiotemporalAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatiotemporalAttention, self).__init__()

#         # Spatial attention (2D convolution)
#         self.spatial_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=7, stride=1, padding=3)

#         # Temporal attention (1D convolution)
#         self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, image):
#         # image: B, C, T, X, Y
#         B, C, T, X, Y = image.shape

#         #### Spatial Attention ####
#         avgout = torch.mean(image, dim=1, keepdim=True)  # Shape: (B, 1, T, X, Y)
#         maxout, _ = torch.max(image, dim=1, keepdim=True)  # Shape: (B, 1, T, X, Y)
#         out = torch.cat([avgout, maxout], dim=1)  # Shape: (B, 2, T, X, Y)

#         # Reshape for 2D convolution: (B * T, 2, X, Y)
#         out = out.permute(0, 2, 1, 3, 4).reshape(B * T, 2, X, Y)
#         out = self.spatial_conv(out)  # Apply 2D convolution on (X, Y)
#         out = self.sigmoid(out)  # Spatial attention weights: (B * T, 1, X, Y)

#         # Reshape back to (B, T, 1, X, Y)
#         out = out.reshape(B, T, 1, X, Y)

#         #### Temporal Attention ####
#         # Reshape for 1D convolution: (B * X * Y, 1, T)
#         out = out.permute(0, 3, 4, 2, 1).reshape(B * X * Y, 1, T)
#         out = self.temporal_conv(out)  # Apply 1D convolution on temporal dimension (T)
#         out = self.sigmoid(out)  # Temporal attention weights: (B * X * Y, 1, T)

#         # Reshape back to (B, X, Y, 1, T), then permute to (B, 1, T, X, Y)
#         out = out.reshape(B, X, Y, 1, T).permute(0, 3, 4, 1, 2).contiguous()

#         return out  # Attention weights for both spatial and temporal dimensions




class SpatiotemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatiotemporalAttention, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        avgout = torch.mean(image, dim=1, keepdim=True)
        maxout, _ = torch.max(image, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out
