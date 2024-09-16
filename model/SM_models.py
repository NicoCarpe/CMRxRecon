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
import torch.utils.checkpoint as cp

# Transpose blocks were not used in this code originally, so make sure the change is necessary
from model.CNN import ConvBlock3D, TransposeConvBlock3D

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
        max_img_size: list
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

        
        masked_kspace_padded, original_size = utils.pad_to_max_size(masked_kspace, max_img_size)
        ACS_mask_padded, _ = utils.pad_to_max_size(ACS_mask, max_img_size)
        # Apply the padded ACS mask to the padded k-space data
        ACS_kspace_padded = ACS_mask_padded * masked_kspace_padded
        # Convert to image space
        ACS_images_padded = utils.ifft2(ACS_kspace_padded)
        
        # Estimate sensitivities independently
        B, C, T, X, Y = ACS_images_padded.shape
        batched_channels = ACS_images_padded.reshape(B * C, 1, T, X, Y)  # Keep channels separated

        sensitivity, gate = self.norm_net(batched_channels)

        gate = gate.reshape(B, C, T, X, Y)
        sensitivity = sensitivity.reshape(B, C, T, X, Y)
        sensitivity = sensitivity / (utils.rss(sensitivity) + 1e-12)
        sensitivity = gate * sensitivity

        # NOTE: we will unpad in our model later
        # Unpad the outputs to the original size
        # sensitivity = utils.unpad_from_max_size(sensitivity, original_size)
        # gate = utils.unpad_from_max_size(gate, original_size)

        return sensitivity, gate
    

#########################################################################################################
# 3D UNet 
#########################################################################################################

class UNet3D(nn.Module):
    """
    PyTorch implementation of a 3D U-Net model from "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        - authours: Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger
        - arXiv:1606.06650
    """
    def __init__(
            self,  
            in_chans: int = 2,
            out_chans: int = 2,
            chans: int = 32, 
            num_pool_layers: int = 4
    ):
        super().__init__()        
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers

        # Initialize the downsampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock3D(self.in_chans, chans)])

        ch = chans

        for i in range(self.num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2))
            ch *= 2

        # Bottleneck layer
        self.bottleneck = ConvBlock3D(ch, ch * 2)

        # Initialize the upsampling path
        self.up_sample_layers = nn.ModuleList()
        self.up_conv_layers = nn.ModuleList()

        for _ in range(self.num_pool_layers - 1):
            self.up_sample_layers.append(TransposeConvBlock3D(ch * 2, ch))
            self.up_conv_layers.append(ConvBlock3D(ch * 2, ch))
            ch //= 2

        self.final_conv = ConvBlock3D(ch * 2, ch)
        self.norm_conv = nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)

        # Gated Convolution
        self.gate_conv = nn.Conv3d(self.out_chans, 1, kernel_size=1, stride=1)
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

        # Learnable Gaussian std
        self.stds = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_pool_layers)])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, in_chans, T, X, Y)`.
        """
        stack = []
        output = image

        # Downsampling path
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        # Bottleneck
        output = self.bottleneck(output)
        cnt = 0

            
        # Upsampling path
        for up_sample, up_conv in zip(self.up_sample_layers, self.up_conv_layers):
            sigma_value = torch.clamp(self.stds[cnt], min=1e-9, max=7)
            sigma = (0, sigma_value.item(), sigma_value.item())  # No smoothing on the temporal dimension
            output_ = utils.gaussian_smooth_3d(output.view(-1, 1, *output.shape[-3:]), sigma=sigma)
            
            output = output_.view(output.shape)
            downsample_layer = stack.pop()
            output = up_sample(output)
            
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:  # Y dimension
                padding[0] = 1  # padding right (Y dimension)
            if output.shape[-2] != downsample_layer.shape[-2]:  # X dimension
                padding[2] = 1  # padding bottom (X dimension)
            if output.shape[-3] != downsample_layer.shape[-3]:  # T dimension
                padding[4] = 1  # padding in the temporal dimension

            if any(padding):
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = up_conv(output)
            
            cnt += 1

        output = self.final_conv(output)

        # Final Gaussian smoothing and norm_conv
        sigma_value = torch.clamp(self.stds[cnt], min=1e-9, max=7)
        sigma = (0, sigma_value.item(), sigma_value.item())  # No smoothing on the temporal dimension
        output_ = utils.gaussian_smooth_3d(output.view(-1, 1, *output.shape[-3:]), sigma=sigma)
        output = output_.view(output.shape)
        output = F.interpolate(output, scale_factor=(2**(self.num_pool_layers-3)))
        output = self.norm_conv(output)

        # Gated Convolution
        gate = torch.sigmoid(self.scale * (self.gate_conv(output) + self.shift))

        return output, gate


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
        
        self.NormNet3D = UNet3D(
            chans=chans,
            num_pool_layers=num_steps,
        )


    def complex_to_chan_dim(self, image: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(image)
        return torch.cat([image.real, image.imag], dim=1) # Use dim=1 for the channel dimension



    def chan_dim_to_complex(self, image: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(image)
        _, C, _, _, _ = image.shape
        assert C % 2 == 0
        C = C // 2
        return torch.complex(image[:, :C], image[:, C:])  # Use dim=1 for the channel dimension



    def norm(self, image: torch.Tensor):
        # group norm
        B, C, T, X, Y = image.shape
        assert C % 2 == 0
        image = image.view(B, 2, C // 2 * T * X * Y)

        mean = image.mean(dim=2).view(B, 2, 1)
        std = image.std(dim=2).view(B, 2, 1)

        image = (image - mean) / (std + 1e-12)

        return image.view(B, C, T, X, Y), mean, std


    def unnorm(self, image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        B, C, T, X, Y = image.shape
        assert C % 2 == 0
        image = image.contiguous().view(B, 2, C // 2 * T * X * Y)
        image = image * std + mean
        return image.view(B, C, T, X, Y)


    def pad(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], List[int], int, int, int]]:
        _, _, T, X, Y = image.shape

        # Calculate the multiples to align to the nearest multiple of 16
        T_mult = ((T - 1) | 15) + 1
        X_mult = ((X - 1) | 15) + 1
        Y_mult = ((Y - 1) | 15) + 1

        # Calculate padding required for each dimension
        T_pad = [math.floor((T_mult - T) / 2), math.ceil((T_mult - T) / 2)]
        X_pad = [math.floor((X_mult - X) / 2), math.ceil((X_mult - X) / 2)]
        Y_pad = [math.floor((Y_mult - Y) / 2), math.ceil((Y_mult - Y) / 2)]

        # Apply padding
        image = F.pad(image, Y_pad + X_pad + T_pad)

        return image, (T_pad, X_pad, Y_pad, T_mult, X_mult, Y_mult)


    def unpad(
        self,
        image: torch.Tensor,
        T_pad: List[int],
        X_pad: List[int],
        Y_pad: List[int],
        T_mult: int,
        X_mult: int,
        Y_mult: int,
    ) -> torch.Tensor:
        return image[..., T_pad[0]: T_mult - T_pad[1], X_pad[0]: X_mult - X_pad[1], Y_pad[0]: Y_mult - Y_pad[1]]


    def forward(
        self, 
        image: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        assert len(image.shape) == 5
        assert torch.is_complex(image)

        # Convert to real-valued tensor and normalize
        image = self.complex_to_chan_dim(image)
        image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)
            
        # Pass through UNet3D (NormNet3D)
        image, gate = self.NormNet3D(image)
        gate = self.unpad(gate, *pad_sizes)
        
        # Unnormalize and convert back to complex
        image = self.unpad(image, *pad_sizes)
        image = self.unnorm(image, mean, std)
        image = self.chan_dim_to_complex(image)

        return image, gate
    

#########################################################################################################
# 3D Convolution 
#########################################################################################################

class ConvBlock3D(nn.Module):
    """
    A Convolutional Block that consists of a convolution layer followed by
    instance normalization and LeakyReLU activation.
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
            nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, in_chans, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, X, Y)`.
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
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, C, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, out_chans, T*2, X*2, Y*2)`.
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

    def __init__(self, in_chans=2, conv_num=0, out_chans=None, max_chans=None):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans_max = max_chans or in_chans
        self.out_chans = out_chans or in_chans

        self.layers = []
        self.layers.append(nn.Sequential(
            nn.Conv3d(self.in_chans, self.chans_max, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(self.chans_max),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ))

        #### Spatiotemporal Attention ####
        self.STA = SpatiotemporalAttention(in_channels=self.in_chans)

        for index in range(conv_num):
            if index == conv_num - 1:
                self.layers.append(nn.Sequential(
                    nn.Conv3d(self.chans_max, self.out_chans, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(self.in_chans),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv3d(self.chans_max, self.chans_max, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(self.chans_max),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ))

        self.body = nn.Sequential(*self.layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, C, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, X, Y)`.
        """
        output = self.body(image)
        output = output + image[:, :2, :, :, :]  # Adjusted to C first dimension
        
        #### Spatiotemporal Attention ###
        output = self.STA(output) * output

        return output

#########################################################################################################
# 2D+t SM Convolution w/ Spatiotemporal Attention (Modified)
#########################################################################################################

class ConvBlockSM2D_t(nn.Module):
    """
    A Convolutional Block that consists of convolution layers each followed by
    instance normalization, LeakyReLU activation, and spatiotemporal attention.
    """
    def __init__(self, in_chans=2, conv_num=0, out_chans=None, max_chans=None):
        super().__init__()

        self.in_chans = in_chans
        self.chans_max = max_chans or in_chans
        self.out_chans = out_chans or in_chans

        # List to store the blocks with both spatial and temporal convolutions
        self.layers = nn.ModuleList()

        for index in range(conv_num):
            current_out_chans = self.out_chans if index == conv_num - 1 else self.chans_max
            self.layers.append(self._create_conv_block(self.chans_max if index > 0 else in_chans, current_out_chans))

        # Spatiotemporal Attention
        self.STA = SpatiotemporalAttention(in_channels=self.in_chans)

    def _create_conv_block(self, in_channels, out_channels):
        """
        Helper function to create a block of spatial and temporal convolutions.
        Each block applies a spatial convolution followed by a temporal convolution,
        then normalization and activation for both.
        """
        return nn.Sequential(
            # Spatial Convolution (2D)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Temporal Convolution (1D)
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image):
        # image: B, C, T, X, Y
        B, C, T, X, Y = image.shape
        input_image = image

        for layer in self.layers:
            # Apply spatial convolution (2D)
            image = image.permute(0, 2, 1, 3, 4).reshape(B * T, C, X, Y)  # Reshape to (B * T, C, X, Y)
            image = layer[0](image)  # Apply Conv2d
            image = layer[1](image)  # InstanceNorm2d
            image = layer[2](image)  # LeakyReLU
            image = image.reshape(B, T, C, X, Y).permute(0, 2, 1, 3, 4)  # Reshape back to (B, C, T, X, Y)

            # Apply temporal convolution (1D)
            image = image.permute(0, 3, 4, 1, 2).reshape(B * X * Y, C, T)  # Reshape to (B * X * Y, C, T)
            image = layer[3](image)  # Apply Conv1d
            image = layer[4](image)  # InstanceNorm1d
            image = layer[5](image)  # LeakyReLU
            image = image.reshape(B, X, Y, C, T).permute(0, 3, 4, 1, 2).contiguous()  # Reshape back to (B, C, T, X, Y)

        # Skip connection - Add the input to the output
        image = image + input_image[:, :2, :, :, :]  # Adjusting to add only the first 2 channels of the input

        # Apply spatiotemporal attention
        image = self.STA(image) * image  

        return image


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

class SpatiotemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatiotemporalAttention, self).__init__()

        # Spatial attention (2D convolution)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=7, stride=1, padding=3)

        # Temporal attention (1D convolution)
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        # image: B, C, T, X, Y
        B, C, T, X, Y = image.shape

        #### Spatial Attention ####
        avgout = torch.mean(image, dim=1, keepdim=True)  # Shape: (B, 1, T, X, Y)
        maxout, _ = torch.max(image, dim=1, keepdim=True)  # Shape: (B, 1, T, X, Y)
        out = torch.cat([avgout, maxout], dim=1)  # Shape: (B, 2, T, X, Y)

        # Reshape for 2D convolution: (B * T, 2, X, Y)
        out = out.permute(0, 2, 1, 3, 4).reshape(B * T, 2, X, Y)
        out = self.spatial_conv(out)  # Apply 2D convolution on (X, Y)
        out = self.sigmoid(out)  # Spatial attention weights: (B * T, 1, X, Y)

        # Reshape back to (B, T, 1, X, Y)
        out = out.reshape(B, T, 1, X, Y)

        #### Temporal Attention ####
        # Reshape for 1D convolution: (B * X * Y, 1, T)
        out = out.permute(0, 3, 4, 2, 1).reshape(B * X * Y, 1, T)
        out = self.temporal_conv(out)  # Apply 1D convolution on temporal dimension (T)
        out = self.sigmoid(out)  # Temporal attention weights: (B * X * Y, 1, T)

        # Reshape back to (B, X, Y, 1, T), then permute to (B, 1, T, X, Y)
        out = out.reshape(B, X, Y, 1, T).permute(0, 3, 4, 1, 2).contiguous()

        return out  # Attention weights for both spatial and temporal dimensions




# class SpatiotemporalAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatiotemporalAttention, self).__init__()
#         self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, image):
#         avgout = torch.mean(image, dim=1, keepdim=True)
#         maxout, _ = torch.max(image, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv3d(out))
#         return out
