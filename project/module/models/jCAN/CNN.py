"""
Oct 12, 2021
Combined and modified by Kaicong Sun <sunkc@shanghaitech.edu.cn>

Jul 23, 2024
Extended to 2D+t by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""

import math
from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
import utils

#########################################################################################################
# 3D UNet
#########################################################################################################

class Unet3D(nn.Module):
    """
    PyTorch implementation of a U-Net model for 2D+t data.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
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
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_chans, chans)])
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

        self.up_conv.append(
            nn.Sequential(
                ConvBlock3D(ch * 2, ch),
                nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, T, H, W)`.
        """
        assert not torch.is_complex(image)
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for up_conv, conv in zip(self.upsampling_conv, self.up_conv):
            downsample_layer = stack.pop()

            output = up_conv(output)
            output = F.interpolate(output, scale_factor=(1, 2, 2))

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

        self.NormNet3D = Unet3D(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,  # 32
            num_pool_layers=num_pool_layers
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
        b, c, t, h, w = x.shape
        assert c % 2 == 0
        x = x.view(b, 2, c // 2 * t * h * w)

        mean = x.mean(dim=2).view(b, 2, 1)
        std = x.std(dim=2).view(b, 2, 1)

        x = (x - mean) / (std + 1e-12)

        return x.view(b, c, t, h, w), mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        b, c, t, h, w = x.shape
        assert c % 2 == 0
        x = x.contiguous().view(b, 2, c // 2 * t * h * w)
        x = x * std + mean
        return x.view(b, c, t, h, w)

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

        # get shapes for unet and normalize
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        #### standard ####
        x = self.NormNet3D(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)

        return x

#########################################################################################################
# 3D Covolution 
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
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, T, H, W)`.
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
            image: Input 5D tensor of shape `(N, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, T, H, W)`.
        """
        x_img = self.layers(image)
        x_gate = self.gatedlayers(image)
        x = x_img * x_gate
        return x

#########################################################################################################
# 3D SM Convolution w/ Spatial Attention
#########################################################################################################

class ConvBlockSM3D(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation, and dropout.
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

        #### Spatial Attention ####
        self.SA = utils.SpatialAttention()

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
            image: Input 5D tensor of shape `(N, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, T, H, W)`.
        """
        output = self.body(image)
        output = output + image[:, :2, :, :, :]
        #### Spatial Attention ###
        output = self.SA(output) * output

        return output

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
            nn.ConvTranspose3d(
                in_chans, out_chans, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False
            ),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 5D tensor of shape `(N, in_chans, T, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, T, H*2, W*2)`.
        """
        return self.layers(image)

