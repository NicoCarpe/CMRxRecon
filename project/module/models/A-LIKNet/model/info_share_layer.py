import torch
import torch.nn as nn
import torch.nn.functional as F
from mri_no_mask import MulticoilForwardOp, MulticoilAdjointOp

class Scalar(nn.Module):
    def __init__(self, init=1.0, train_scale=1.0, name=None):
        super(Scalar, self).__init__()
        self.init = init
        self.train_scale = train_scale
        self.weight = nn.Parameter(torch.tensor(self.init))

    def forward(self, inputs):
        weight = F.relu(self.weight) * self.train_scale
        return inputs * weight

class InfoShareLayer(nn.Module):
    def __init__(self, name='InfoShareLayer'):
        super(InfoShareLayer, self).__init__()
        # two scalar for add operation
        self.tau_ksp = Scalar(init=0.5)
        self.tau_img = Scalar(init=0.5)
        self.multicoil_forward_op = MulticoilForwardOp(center=True)
        self.multicoil_adjoint_op = MulticoilAdjointOp(center=True)

    def forward(self, kspace, image, mask, smap):
        # transform image to kspace
        img_fft = self.multicoil_forward_op(image, mask, smap)
        ksp_1 = kspace
        ksp_2 = img_fft

        # weighted combined k-space
        weighted_ksp_1 = self.tau_ksp(ksp_1)
        ksp_2_weight = 1.0 - self.tau_ksp.weight
        weighted_ksp_2 = ksp_2 * ksp_2_weight
        new_ksp = weighted_ksp_1 + weighted_ksp_2

        # transform kspace to image
        ksp_ifft = self.multicoil_adjoint_op(kspace, mask, smap)
        img_1 = image
        img_2 = ksp_ifft

        # weighted combined image
        weighted_img_1 = self.tau_img(img_1)
        img_2_weight = 1.0 - self.tau_img.weight
        weighted_img_2 = img_2 * img_2_weight
        new_img = weighted_img_1 + weighted_img_2

        return new_ksp, new_img
