import torch
import torch.nn as nn
import torch.nn.functional as F

from ksp_net import KspNetAttention
from info_share_layer import InfoShareLayer
from low_rank_net import Scalar, ComplexAttentionLSNet
from mri import MulticoilForwardOp, MulticoilAdjointOp

class DCLayer(nn.Module):
    def __init__(self, A, AH):
        super(DCLayer, self).__init__()
        self.A = A
        self.AH = AH

    def forward(self, x, mask, sub_y):
        masked_pred_ksp = x * mask
        scaled_pred_ksp = self.A(masked_pred_ksp)
        yu_weight = 1.0 - self.AH.weight
        scaled_sampled_ksp = sub_y * yu_weight
        other_points = x * (1 - mask)
        out = scaled_pred_ksp + scaled_sampled_ksp + other_points
        return out

class A_LIKNet(nn.Module):
    def __init__(self, num_iter=8, name='A_LIKNet'):
        super(A_LIKNet, self).__init__()

        self.S_end = num_iter

        self.ISL = nn.ModuleList()
        self.KspNet = nn.ModuleList()
        self.ImgLrNet = nn.ModuleList()
        self.ImgDC = nn.ModuleList()

        self.ksp_dc_weight = Scalar(init=0.5)

        for _ in range(self.S_end):
            self.ISL.append(InfoShareLayer())
            self.KspNet.append(KspNetAttention(output_filter=25))
            self.ImgLrNet.append(ComplexAttentionLSNet())
            self.ImgDC.append(DCLayer(A=MulticoilForwardOp(center=True), AH=MulticoilAdjointOp(center=True)))

    def ksp_dc(self, kspace, mask, sub_ksp):
        masked_pred_ksp = kspace * mask
        scaled_pred_ksp = self.ksp_dc_weight(masked_pred_ksp)
        yu_weight = 1.0 - self.ksp_dc_weight.weight
        scaled_sampled_ksp = sub_ksp * yu_weight
        other_points = kspace * (1 - mask)
        out = scaled_pred_ksp + scaled_sampled_ksp + other_points
        return out

    def update_xy(self, x, y, i, num_iter, constants):
        sub_y, mask, smap = constants

        # kspace network
        ksp_net = self.KspNet[i]
        y = ksp_net(y)

        # image and low-rank network
        img_lr_net = self.ImgLrNet[i]
        x = img_lr_net(x, num_iter)

        # dc operation
        y = self.ksp_dc(y, mask, sub_y)
        img_dc_layer = self.ImgDC[i]
        x = img_dc_layer(x, mask, sub_y)

        # information sharing
        info_share_layer = self.ISL[i]
        y, x = info_share_layer(y, x, mask, smap)

        return x, y

    def forward(self, inputs):
        x, y, mask, smaps = inputs
        constants = inputs[1:]
        for i in range(self.S_end):
            x, y = self.update_xy(x, y, i, num_iter=self.S_end, constants=constants)
        return y, x
