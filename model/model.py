# -*- coding: utf-8 -*-
"""
@author: sunkg

Jul 25, 2024
Extended by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""
import torch
import torch.fft
import torch.nn.functional as F
from basemodel import BaseModel
import metrics
import torch.nn as nn
import fD2RT3D
from utils import rss, fft2, ifft2, ssimloss
import utils
import numpy as np

def generate_rhos(num_recurrent):
    rhos = [0.85 ** i for i in range(num_recurrent - 1, -1, -1)]
    return rhos

# Modified TV loss to consider smoothness in the temporal dimension as well as the spatial dimensions 
def TV_loss(img, weight):
    bs_img, t_img, c_img, h_img, w_img = img.size()
    tv_t = torch.pow(img[:, 1:, :, :, :] - img[:, :-1, :, :, :], 2).sum()
    tv_h = torch.pow(img[:, :, :, 1:, :] - img[:, :, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, :, 1:] - img[:, :, :, :, :-1], 2).sum()
    return (weight * (tv_h + tv_w + tv_t)) / (bs_img * t_img * c_img * h_img * w_img)

class ReconModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rhos = generate_rhos(self.cfg.num_recurrent)
        self.device = self.cfg.device
       
        self.net_R = fD2RT3D.fD2RT3D(
                        coils=self.cfg.coils,
                        num_heads=self.cfg.num_heads, 
                        window_size=self.cfg.window_size, 
                        depths=self.cfg.depths, 
                        patch_size=self.cfg.patch_size, 
                        embed_dim=self.cfg.embed_dim, 
                        mlp_ratio=self.cfg.mlp_ratio, 
                        qkv_bias=True, 
                        qk_scale=None, 
                        drop=0., 
                        attn_drop=0., 
                        drop_path=0., 
                        norm_layer=nn.LayerNorm, 
                        n_SC=self.cfg.n_SC, 
                        num_recurrent=self.cfg.num_recurrent, 
                        sens_chans=self.cfg.sens_chans, 
                        sens_steps=self.cfg.sens_steps, 
                        ds_ref=self.cfg.ds_ref
                    ).to(self.device)

    def set_input_noGT(self, masked_kspace, mask):
        B, T, C, H, W = masked_kspace.shape
        self.Target_f_rss = torch.zeros([B, T, 1, H, W], dtype=torch.complex64)
        self.Target_Kspace_f = torch.zeros([B, T, C, H, W], dtype=torch.complex64)
        self.Target_Kspace_sampled = masked_kspace
        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))
        self.mask_tar = mask

    def set_input_GT(self, masked_kspace, mask, target):
        self.Target_f_rss = rss(target)
        self.Target_Kspace_f = fft2(target)
        self.mask_tar = mask
        self.Target_Kspace_sampled = self.Target_Kspace_f * self.mask_tar
        self.Target_sampled_rss = rss(ifft2(self.Target_Kspace_sampled))

    def forward(self, masked_kspace, mask, num_low_frequencies, max_img_size, target=None):
        if target is not None:
            self.set_input_GT(masked_kspace, mask, target)
        else:
            self.set_input_noGT(masked_kspace, mask)

        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            self.recs_complex, self.rec_rss, self.sens_maps, self.rec_img = self.net_R(
                Target_Kspace_u=self.Target_Kspace_sampled,
                mask=self.mask_tar,
                num_low_frequencies=num_low_frequencies,
                max_img_size=max_img_size
            )

            # Loss computation
            self.loss_all = 0
            self.loss_fidelity = 0
            self.local_fidelities = []
            for i in range(self.cfg.num_recurrent):
                loss_fidelity = F.l1_loss(rss(self.recs_complex[i]), self.Target_f_rss) + self.cfg.lambda0 * F.l1_loss(utils.sens_expand(self.recs_complex[i], self.sens_maps), self.Target_Kspace_f)
                self.local_fidelities.append(self.rhos[i] * loss_fidelity)
                self.loss_fidelity += self.local_fidelities[-1]

            self.loss_all += self.loss_fidelity
            self.loss_consistency = self.cfg.lambda1 * F.l1_loss(self.mask_tar * utils.sens_expand(self.rec_img, self.sens_maps), self.Target_Kspace_sampled)
            self.loss_all += self.loss_consistency
            self.loss_TV = TV_loss(torch.abs(self.sens_maps), self.cfg.lambda3)
            self.loss_all += self.loss_TV
            self.loss_ssim = self.cfg.lambda2 * ssimloss(self.rec_rss, self.Target_f_rss)
            self.loss_all += self.loss_ssim

        return self.local_fidelities, self.loss_fidelity, self.loss_consistency, self.loss_TV, self.loss_ssim, self.loss_all
    
    def test(self, Target_img_f):
        assert not self.training
        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            with torch.no_grad():
                self.forward(Target_img_f)
                
                self.metric_PSNR = metrics.psnr(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_SSIM = metrics.ssim(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MAE = metrics.mae(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_MSE = metrics.mse(self.Target_f_rss, self.rec_rss, self.cfg.GT)
                self.metric_PSNR_raw = metrics.psnr(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                self.metric_SSIM_raw = metrics.ssim(self.Target_f_rss, self.Target_sampled_rss, self.cfg.GT)
                
                self.Eval = {
                    'PSNR': self.metric_PSNR,
                    'SSIM': self.metric_SSIM
                }

    def prune(self, *args, **kwargs):
        assert False, 'Take care of amp'
        return self.net_mask_tar.prune(*args, **kwargs)

    def get_vis(self, content=None):
        assert content in [None, 'scalars', 'histograms', 'images']
        vis = {}
        if content == 'scalars' or content is None:
            vis['scalars'] = {}
            for loss_name in filter(lambda x: x.startswith('loss_'), self.__dict__.keys()):
                loss_val = getattr(self, loss_name)
                if loss_val is not None:
                    vis['scalars'][loss_name] = loss_val.detach().item()
            for metric_name in filter(lambda x: x.startswith('metric_'), self.__dict__.keys()):
                metric_val = getattr(self, metric_name)
                if metric_val is not None:
                    vis['scalars'][metric_name] = metric_val
        if content == 'images' or content is None:
            vis['images'] = {}
            for image_name in filter(lambda x: x.endswith('_rss'), self.__dict__.keys()):
                image_val = getattr(self, image_name)
                if (image_val is not None) and (image_val.shape[1] == 1 or image_val.shape[1] == 3) and not torch.is_complex(image_val):
                    vis['images'][image_name] = image_val.detach()
        if content == 'histograms' or content is None:
            vis['histograms'] = {}
            if self.net_mask_tar.weight is not None:
                vis['histograms']['weights'] = {'values': self.net_mask_tar.weight.detach()}
        return vis
