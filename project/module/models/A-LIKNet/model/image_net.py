import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class complex_SE_time_layer(nn.Module):
    def __init__(self, time_size, bottle_size=2):
        super(complex_SE_time_layer, self).__init__()
        self.time_size = time_size
        self.SE_time = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, time_size, 1, 1)),
            nn.Linear(time_size, bottle_size * 2),
            nn.Linear(bottle_size * 2, time_size * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_excitation = torch.cat((x.real, x.imag), dim=1)
        x_excitation = self.SE_time(x_excitation)
        x_excitation = x_excitation.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x = torch.complex(x.real * x_excitation[:, :self.time_size, :, :, :],
                          x.imag * x_excitation[:, self.time_size:, :, :, :])
        return x

class UNet_2Dt(nn.Module):
    def __init__(self, dim='2Dt', filters=64, kernel_size_2d=(1, 3, 3), kernel_size_t=(3, 1, 1), pool_size=2,
                 num_layer_per_level=2, num_level=4, activation='relu', activation_last='relu',
                 kernel_size_last=1, use_bias=True, normalization='none', downsampling='mp', upsampling='tc',
                 padding='none', **kwargs):
        super(UNet_2Dt, self).__init__()

        # Initialize attributes
        self.dim = dim
        self.filters = filters
        self.kernel_size_2d = kernel_size_2d
        self.kernel_size_t = kernel_size_t
        self.pool_size = pool_size
        self.num_layer_per_level = num_layer_per_level
        self.num_level = num_level
        self.activation = activation
        self.activation_last = activation_last
        self.kernel_size_last = kernel_size_last
        self.use_bias = use_bias
        self.normalization = normalization
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.padding = padding
        self.norm_layer = nn.InstanceNorm3d
        self.activation_layer = nn.ReLU
        self.SE_time = complex_SE_time_layer
        self.use_padding = False
        self.ops = []

    def create_layers(self, **kwargs):
        self.ops = []
        # Encoder
        stage = []
        for ilevel in range(self.num_level):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_2d, **kwargs))
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_t, **kwargs))
            if self.downsampling == 'mp':
                level.append(nn.MaxPool3d(self.pool_size))
            else:
                level.append(nn.Conv3d(self.filters * (2 ** ilevel), self.filters * (2 ** ilevel), (1, 1, 1),
                                       stride=2, padding='same'))
            stage.append(level)
        self.ops.append(stage)

        # Bottleneck
        stage = []
        for ilayer in range(self.num_layer_per_level):
            stage.append(self.conv_layer(self.filters * (2 ** self.num_level), self.kernel_size_2d, **kwargs))
            stage.append(self.conv_layer(self.filters * (2 ** self.num_level), self.kernel_size_t, **kwargs))
        if self.upsampling == 'us':
            stage.append(nn.Upsample(scale_factor=self.pool_size))
        elif self.upsampling == 'tc':
            stage.append(nn.ConvTranspose3d(self.filters * (2 ** (self.num_level - 1)), self.filters * (2 ** (self.num_level - 1)),
                                            (1, 1, 1), stride=self.pool_size, padding='same'))
        self.ops.append(stage)

        # Decoder
        stage = []
        for ilevel in range(self.num_level - 1, -1, -1):
            level = []
            for ilayer in range(self.num_layer_per_level):
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_t, **kwargs))
                level.append(self.conv_layer(self.filters * (2 ** ilevel), self.kernel_size_2d, **kwargs))
            if ilevel == 1:
                level.append(self.SE_time(time_size=14))
            if ilevel == 0:
                level.append(self.SE_time(time_size=28))
            if ilevel > 0:
                if self.upsampling == 'us':
                    level.append(nn.Upsample(scale_factor=self.pool_size))
                elif self.upsampling == 'tc':
                    level.append(nn.ConvTranspose3d(self.filters * (2 ** (ilevel - 1)), self.filters * (2 ** (ilevel - 1)),
                                                    (1, 1, 1), stride=self.pool_size, padding='same'))
            stage.append(level)
        self.ops.append(stage)

        # Output convolution
        self.ops.append(self.conv_layer(1, self.kernel_size_last, activation=self.activation_last, **kwargs))

    def conv_layer(self, out_channels, kernel_size, **kwargs):
        return nn.Sequential(
            nn.Conv3d(self.filters, out_channels, kernel_size, stride=1, padding='same'),
            self.norm_layer(out_channels),
            self.activation_layer()
        )

    def forward(self, inputs):
        if self.use_padding:
            x = F.pad(inputs, self.pad)
        else:
            x = inputs

        xforward = []
        # Encoder
        for ilevel in range(self.num_level):
            for iop, op in enumerate(self.ops[0][ilevel]):
                if iop == len(self.ops[0][ilevel]) - 1:
                    xforward.append(x)
                if op is not None:
                    x = op(x)

        # Bottleneck
        for op in self.ops[1]:
            if op is not None:
                x = op(x)

        # Decoder
        for ilevel in range(self.num_level - 1, -1, -1):
            x = torch.cat([x, xforward[ilevel]], dim=1)
            for op in self.ops[2][self.num_level - 1 - ilevel]:
                if op is not None:
                    x = op(x)

        # Output convolution
        x = self.ops[3](x)
        if self.use_padding:
            x = F.pad(x, self.pad)
        return x

class ComplexUNet_2Dt(UNet_2Dt):
    def __init__(self, dim='2Dt', filters=64, kernel_size_2d=(1, 3, 3), kernel_size_t=(3, 1, 1), pool_size=2,
                 num_layer_per_level=2, num_level=4, activation='ModReLU', activation_last='ModReLU',
                 kernel_size_last=1, use_bias=True, normalization='none', downsampling='st', upsampling='tc',
                 padding='none', **kwargs):
        super(ComplexUNet_2Dt, self).__init__(dim, filters, kernel_size_2d, kernel_size_t, pool_size,
                                             num_layer_per_level, num_level, activation, activation_last,
                                             kernel_size_last, use_bias, normalization, downsampling,
                                             upsampling, padding, **kwargs)

        self.dim = '3D'
        self.conv_layer = nn.Conv3d
        self.pad_layer = nn.ZeroPad3d if self.padding.lower() == 'zero' else F.pad
        self.crop_layer = nn.Identity()  # No direct equivalent, handle separately if needed

        self.out_cha = 1

        if downsampling == 'mp':
            self.down_layer = nn.MaxPool3d
            self.strides = [1] * num_layer_per_level
        elif downsampling == 'st':
            self.down_layer = nn.Conv3d
            self.strides = [self.pool_size, self.pool_size, self.pool_size]
        else:
            raise RuntimeError(f"Downsampling operation {downsampling} not implemented!")

        if upsampling == 'us':
            self.up_layer = nn.Upsample
        elif upsampling == 'tc':
            self.up_layer = nn.ConvTranspose3d
        else:
            raise RuntimeError(f"Upsampling operation {upsampling} not implemented!")

        super().create_layers(**kwargs)

def callCheck(fhandle, **kwargs):
    return fhandle(**kwargs) if fhandle is not None else fhandle
