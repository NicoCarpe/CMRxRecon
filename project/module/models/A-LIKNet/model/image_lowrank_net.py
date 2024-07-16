import torch
import torch.nn as nn
import torch.nn.functional as F
from image_net import ComplexUNet_2Dt
from low_rank_net import LNet_xyt_Batch

def extract_patches(x):
    # x: (t, x, y, 1)
    nb, nx, ny, nc = x.shape
    b_x = nx % 4 + 4
    b_y = ny % 4 + 4
    patch_x = (nx + 3 * b_x) // 4
    patch_y = (ny + 3 * b_y) // 4
    patches = x.unfold(1, patch_x, patch_x - b_x).unfold(2, patch_y, patch_y - b_y)
    return patches, patch_x, patch_y

def extract_patches_inverse(original_x, patches):
    _x = torch.zeros_like(original_x)
    _y, patch_x, patch_y = extract_patches(_x)
    temp = torch.ones_like(_y, dtype=torch.complex64)
    grad = torch.autograd.grad(_y, _x, grad_outputs=temp, retain_graph=True)[0]
    return torch.autograd.grad(_y, _x, grad_outputs=patches, retain_graph=True)[0] / grad

class Scalar(nn.Module):
    def __init__(self, init=1.0, train_scale=1.0, name=None):
        super(Scalar, self).__init__()
        self.init = init
        self.train_scale = train_scale
        self.weight = nn.Parameter(torch.tensor(self.init))

    def forward(self, inputs):
        weight = F.relu(self.weight) * self.train_scale
        return inputs * weight

def get_complex_attention_CNN():
    return ComplexUNet_2Dt(dim='2Dt', filters=12, kernel_size_2d=(1, 5, 5), kernel_size_t=(3, 1, 1), downsampling='mp',
                           num_level=2, num_layer_per_level=1, activation_last=None)

class ComplexAttentionLSNet(nn.Module):
    def __init__(self, name='ComplexAttentionImageLRNet'):
        super(ComplexAttentionLSNet, self).__init__()
        # define CNN block (sparse regularizer)
        self.R = get_complex_attention_CNN()
        # define low rank operator D
        self.D = LNet_xyt_Batch(num_patches=80)
        # denoiser strength
        self.tau = Scalar(init=0.1, name='tau')
        # image branch weight
        self.p_weight = Scalar(init=0.5, name='p_weight')

    def reshape_patch(self, patch, patch_x, patch_y):
        nb = patch.shape[0]
        patches_stacked = patch.view(nb, 4, 4, patch_x, patch_y)  # (25, 4, 4, px, py)
        patches_stacked = patches_stacked.view(1, 25, 16, patch_x, patch_y)  # (1, 25, 16, px, py)
        return patches_stacked

    def reshape_xyt_patch(self, patch):
        patches_stacked_split = patch.permute(0, 2, 1, 3, 4)  # (5, 16, t, x, y)
        nb_t, nb_xy, pt, px, py = patches_stacked_split.shape
        patches_stacked_split = patches_stacked_split.view(nb_t * nb_xy, pt, px, py)  # (80, pt, px, py)
        return patches_stacked_split

    def split_time(self, x):
        nt = x.shape[1]
        split_num = 5  # temporal patch size
        interval = nt // split_num
        x_intervals = [x[:, i * interval:(i + 1) * interval, :, :, :] for i in range(split_num)]
        x_new = torch.cat(x_intervals, dim=0)
        return x_new

    def recover_time_sequence(self, x):
        nb = x.shape[0]
        x_intervals = [x[i, :, :, :, :] for i in range(nb)]
        new_x = torch.cat(x_intervals, dim=0)
        x = new_x.unsqueeze(0)
        return x

    def recover_xyt_to_xy_patch(self, patch):
        nb, pt, px, py = patch.shape
        patches_stacked_split = patch.view(5, 16, pt, px, py)  # (5, 16, t, x, y)
        patches_stacked_split = patches_stacked_split.permute(0, 2, 1, 3, 4)  # (5, 5, 16, x, y)
        patches_stacked = self.recover_time_sequence(patches_stacked_split)  # (1, 25, 16, x, y)
        patches_stacked = patches_stacked.view(25, 4, 4, px, py)
        patches = patches_stacked.view(25, 4, 4, px * py)
        return patches

    def forward(self, image, num_iter):
        x = image

        # denoiser operation
        den = self.R(x)
        p = x - self.tau(den) / num_iter

        # low rank operation (D)
        patches, patch_x, patch_y = extract_patches(x.squeeze(0))
        x_patches_stacked = self.reshape_patch(patches, patch_x, patch_y)  # (1, 25, 16, patch_x, patch_y)
        x_patches_stacked_split = self.split_time(x_patches_stacked)  # (5, 5, 16, patch_x, patch_y)
        x_patches_stacked_split = self.reshape_xyt_patch(x_patches_stacked_split)  # (80, 5, px, py)
        q_patches_stacked_split = self.D(x_patches_stacked_split)  # (80, 5, px, px)
        q_patches = self.recover_xyt_to_xy_patch(q_patches_stacked_split)
        q = extract_patches_inverse(x.squeeze(0), q_patches)
        q = q.unsqueeze(0)

        # weighted combination
        weighted_p = self.p_weight(p)
        q_weight = 1.0 - self.p_weight.weight
        weighted_q = q * q_weight
        x = weighted_p + weighted_q

        return x
