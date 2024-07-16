import torch
import torch.nn as nn
import torch.nn.functional as F

class complex_SE_coil_layer(nn.Module):
    def __init__(self, coil_size, bottle_size=2):
        super(complex_SE_coil_layer, self).__init__()
        self.coil_size = coil_size
        self.SE_coil = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, None, coil_size)),
            nn.Linear(coil_size, bottle_size * 2),
            nn.Linear(bottle_size * 2, coil_size * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_excitation = torch.cat((x.real, x.imag), dim=-1)
        x_excitation = self.SE_coil(x_excitation)
        x_excitation = x_excitation.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x = torch.complex(x.real * x_excitation[:, :, :, :, :self.coil_size],
                          x.imag * x_excitation[:, :, :, :, self.coil_size:])
        return x

class KspNetAttention(nn.Module):
    def __init__(self, output_filter=25, activation='ModReLU', use_bias=False):
        super(KspNetAttention, self).__init__()
        self.dilation_rate = 1
        self.use_bias = use_bias
        self.activation = nn.ReLU() if activation == 'ModReLU' else None
        self.SE_coil = complex_SE_coil_layer

        self.Nw = nn.ModuleList()
        self.Nw.append(nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(5, 5, 3),
                                 dilation=self.dilation_rate, padding='same', bias=self.use_bias))
        self.Nw.append(self.SE_coil(coil_size=15))

        # second layer
        self.Nw.append(nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(5, 5, 3),
                                 dilation=self.dilation_rate, padding='same', bias=self.use_bias))
        self.Nw.append(self.SE_coil(coil_size=15))

        # output layer
        self.Nw.append(nn.Conv3d(in_channels=8, out_channels=output_filter, kernel_size=(3, 3, 3),
                                 dilation=self.dilation_rate, padding='same', bias=self.use_bias))

    def forward(self, inputs):
        x = inputs
        for op in self.Nw:
            x = op(x)
        return x
