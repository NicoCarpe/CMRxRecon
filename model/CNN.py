# -*- coding: utf-8 -*-
"""
Oct 12, 2021
Combined and modified by Kaicong Sun <sunkc@shanghaitech.edu.cn>

2024
Extended to 3D by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""

import math
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from model import utils

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
        in_chans: int = 20,
        out_chans: int = 20,
        chans: int = 16,  # 32
        num_pool_layers: int = 4,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers

        # Initialize the downsampling path
        self.down_sample_layers = nn.ModuleList([ConvBlock3D(self.in_chans, chans)])

        ch = chans

        for _ in range(self.num_pool_layers - 1):
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

        self.up_sample_layers.append(TransposeConvBlock3D(ch * 2, ch))

        self.output_layer = TransposeConvBlock3D(ch * 2, ch)
        self.output_conv = nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1) 

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, C, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, X, Y)`.
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

        # Upsampling path
        for up_sample, up_conv in zip(self.up_sample_layers, self.up_conv_layers):
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

        output = self.output_layer(output)
        output = self.output_conv(output)

        return output


#########################################################################################################
# 3D NormNet
#########################################################################################################

class NormNet3D(nn.Module):
    """
    Normalized Net model: in Unet or ResNet

    This is the same as a regular 3D U-Net, but with normalization applied to the
    input before the 3D U-Net. This keeps the values more numerically stable
    during training.

    Note NormUnet is designed for complex input/output only.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,  # 32
        num_pool_layers: int = 3,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the complex input.
            out_chans: Number of channels in the complex output.
        """
        super().__init__()

        self.NormNet3D = UNet3D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,  # 32
            num_pool_layers=num_pool_layers
        )

    def complex_to_chan_dim(self, image: torch.Tensor) -> torch.Tensor:
        assert torch.is_complex(image)
        return torch.cat([image.real, image.imag], dim=1)

    def chan_dim_to_complex(self, image: torch.Tensor) -> torch.Tensor:
        assert not torch.is_complex(image)
        _, C, _, _, _ = image.shape
        assert C % 2 == 0
        C = C // 2
        return torch.complex(image[:, :C], image[:, C:])

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
        T_mult = ((T - 1) | 15) + 1
        X_mult = ((X - 1) | 15) + 1
        Y_mult = ((Y - 1) | 15) + 1
        T_pad = [math.floor((T_mult - T) / 2), math.ceil((T_mult - T) / 2)]
        X_pad = [math.floor((X_mult - X) / 2), math.ceil((X_mult - X) / 2)]
        Y_pad = [math.floor((Y_mult - Y) / 2), math.ceil((Y_mult - Y) / 2)]
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

        # get shapes for unet and normalize
        image, mean, std = self.norm(image)
        image, pad_sizes = self.pad(image)

        #### standard ####
        image = self.NormNet3D(image)

        # get shapes back and unnormalize
        image = self.unpad(image, *pad_sizes)
        image = self.unnorm(image, mean, std)

        return image

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
# 3D Gated Convolution
#########################################################################################################

class GatedConvBlock3D(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation, and dropout.
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
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.gatedlayers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(B, C, T, X, Y)`.

        Returns:
            Output tensor of shape `(B, out_chans, T, X, Y)`.
        """
        x_img = self.layers(image)
        x_gate = self.gatedlayers(image)
        x = x_img * x_gate
        return x


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