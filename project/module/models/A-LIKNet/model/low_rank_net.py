import torch
import torch.nn as nn
import torch.nn.functional as F

class LNet_xyt_Batch(nn.Module):
    def __init__(self, num_patches):
        super(LNet_xyt_Batch, self).__init__()
        # initialization of SVD threshold, default value = -2
        self.thres_coef = nn.Parameter(torch.tensor(-2.0).repeat(num_patches))

    def low_rank(self, L):
        L_pre = L  # (nb, nt, nx*ny)
        U, S, V = torch.svd(L_pre)
        # s is a tensor of singular values. shape is [..., P]. (nb, nt)
        # u is a tensor of left singular vectors. shape is [..., M, P]. (nb, nt, nt)
        # v is a tensor of right singular vectors. shape is [..., N, P]. (nb, nx*ny, nt)

        # update the threshold
        # s[..., 0] is the largest value
        thres = torch.sigmoid(self.thres_coef) * S[:, 0]  # shape=(80,)
        thres = thres.unsqueeze(-1)  # shape=(80, 1)

        # Only keep singular values greater than thres
        S = F.relu(S - thres) + thres * torch.sigmoid(S - thres)
        S = torch.diag_embed(S)  # (nb, nt, nt)
        V_conj = V.transpose(-2, -1).conj()  # (nb, nt, nx*ny)

        # U: (nb, nt, nt), S: (nb, nt, nt)
        US = torch.matmul(U, S)  # (nb, nt, nt)
        L = torch.matmul(US, V_conj)  # (nb, nt, nx*ny)
        return L

    def forward(self, inputs):
        # L0: zero-filled image
        # L1~Ln: previous reconstructed images
        image = inputs

        # compress coil dimension
        if len(inputs.shape) == 5:
            image = torch.squeeze(image, axis=-1)
        nb, nt, nx, ny = image.shape
        if nb is None:
            nb = 1
        L_pre = image.view(nb, nt, nx * ny)
        L = self.low_rank(L_pre)

        L = L.view(nb, nt, nx, ny)
        if len(inputs.shape) == 5:
            L = L.unsqueeze(-1)  # (nb, nt, nx, ny, 1)

        return L
