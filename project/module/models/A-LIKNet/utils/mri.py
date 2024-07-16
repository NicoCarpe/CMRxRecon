import torch
import torch.nn as nn

class Smaps(nn.Module):
    def forward(self, img, smaps):
        img = img.to(torch.complex64)
        smaps = smaps.to(torch.complex64)
        return img * smaps

class SmapsAdj(nn.Module):
    def forward(self, coilimg, smaps):
        coilimg = coilimg.to(torch.complex64)
        smaps = smaps.to(torch.complex64)
        return torch.sum(coilimg * torch.conj(smaps), dim=-1)

class MaskKspace(nn.Module):
    def forward(self, kspace, mask):
        return kspace * mask  # Assuming merlintf.complex_scale is equivalent to element-wise multiplication

class ForwardOp(nn.Module):
    def __init__(self, center=False):
        super(ForwardOp, self).__init__()
        self.center = center
        self.mask = MaskKspace()

    def forward(self, image, mask, dims):
        kspace = torch.fft.fft2(image, norm='ortho') if self.center else torch.fft.fft2(image)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class AdjointOp(nn.Module):
    def __init__(self, center=False):
        super(AdjointOp, self).__init__()
        self.center = center
        self.mask = MaskKspace()

    def forward(self, kspace, mask, kshape):
        masked_kspace = self.mask(kspace, mask)
        img = torch.fft.ifft2(masked_kspace, norm='ortho') if self.center else torch.fft.ifft2(masked_kspace)
        return img.unsqueeze(-1)

class MulticoilForwardOp(nn.Module):
    def __init__(self, center=False):
        super(MulticoilForwardOp, self).__init__()
        self.center = center
        self.mask = MaskKspace()
        self.smaps = Smaps()

    def forward(self, image, mask, smaps):
        coilimg = self.smaps(image, smaps)
        kspace = torch.fft.fft2(coilimg, norm='ortho') if self.center else torch.fft.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace

class MulticoilAdjointOp(nn.Module):
    def __init__(self, center=False):
        super(MulticoilAdjointOp, self).__init__()
        self.center = center
        self.mask = MaskKspace()
        self.adj_smaps = SmapsAdj()

    def forward(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = torch.fft.ifft2(masked_kspace, norm='ortho') if self.center else torch.fft.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        return img.unsqueeze(-1)
