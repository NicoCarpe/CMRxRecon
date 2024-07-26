# -*- coding: utf-8 -*-
"""
@author: sunkg

Jul 23, 2024
Modified by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""
import torch
import torch.nn as nn
import SwinTransformer3D
import CNN
import utils
import sensitivity_model

class DDRB(nn.Module):
    def __init__(self, coils_all, img_size, num_heads, window_size, depths, embed_dim, 
                 patch_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, ds_ref=True, scale=0.1):
        super().__init__()

        #### same stepsize for re and im ###
        self.stepsize = nn.Parameter(0.1 * torch.rand(1))

        self.LeakyReLU = nn.LeakyReLU()
        self.img_size = img_size
        self.coils_all = coils_all
        self.ds_ref = ds_ref
        self.scale = scale
        self.CNN = CNN.NormNet3D(in_chans=coils_all * 2, out_chans=coils_all, chans=32)  # using Unet for K-space, 2 is for real and imaginary channels

        

        """
        Swin-T: C = 96,  layer numbers = {2, 2, 6, 2}
        Swin-S: C = 96,  layer numbers = {2, 2, 18, 2} <--
        Swin-B: C = 128, layer numbers = {2, 2, 18, 2}
        Swin-L: C = 192, layer numbers = {2, 2, 18, 2}
        """
        # Replaced ViT with SwinTransformer3D
        self.SwinTransformer3D = SwinTransformer3D.SwinTransformer3D(
            patch_size=patch_size,
            in_chans=2,  # assuming 2 channels for real and imaginary values
            embed_dim=embed_dim,
            depths=depths,  # example depths, modify as needed
            num_heads=num_heads,  # example heads, modify as needed
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path,
            norm_layer=norm_layer,
            patch_norm=ds_ref,
            frozen_stages=-1,
            use_checkpoint=False
        )

    def forward(self, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate):

        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        Target_Kspace_f = utils.complex_to_chan_dim(Target_Kspace_f)
        Target_img_f = utils.complex_to_chan_dim(Target_img_f)

        output_CNN = self.CNN(Target_Kspace_f)

        # Replace ViT forward pass with SwinTransformer3D forward pass
        output_SwinTransformer3D = self.SwinTransformer3D(Target_img_f)

        #### denormalize and turn back to complex values #### 
        output_CNN = utils.chan_dim_to_complex(output_CNN)
        output_SwinTransformer3D = utils.chan_dim_to_complex(output_SwinTransformer3D)
        Target_img_f = utils.chan_dim_to_complex(Target_img_f)

        Target_Kspace_f_down = utils.sens_expand(Target_img_f, sens_maps_updated)
        term1 = 2 * utils.sens_reduce(mask * (mask * Target_Kspace_f_down - Target_Kspace_u), sens_maps_updated)
        term2 = utils.sens_reduce(output_CNN, sens_maps_updated)

        #### update with scaling ####
        Target_img_f = Target_img_f - self.stepsize * (term1 + self.scale * term2 + self.scale * output_SwinTransformer3D)

        return Target_img_f

class fD2RT3D(nn.Module):
    def __init__(self, coils, img_size, num_heads, window_size, depths, patch_size, embed_dim,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, num_recurrent=5, sens_chans=8,
                 sens_steps=4, mask_center=True, ds_ref=True, scale=0.1):
        super().__init__()

        self.sens_net3D = sensitivity_model.SensNet3D(
            chans=sens_chans,
            sens_steps=sens_steps,
            mask_center=mask_center
        )
        self.ds_ref = ds_ref
        self.scale = scale  # scaling layer 
        self.coils = coils
        self.stepsize = nn.Parameter(0.1 * torch.rand(1))
        self.num_recurrent = num_recurrent

        # calls the model for each number of specified unrolls
        self.recurrent = nn.ModuleList([DDRB(self, coils_all=coils, img_size=img_size, num_heads=num_heads, 
                                             window_size=window_size, depths=depths, patch_size=patch_size,
                                             embed_dim=embed_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                             norm_layer=norm_layer, n_SC=n_SC, ds_ref=ds_ref, scale=scale) 
                                        for _ in range(num_recurrent)])

        #### CNN in SMRB ####
        self.ConvBlockSM3D = nn.ModuleList([CNN.ConvBlockSM3D(in_chans=2, conv_num=2) for _ in range(num_recurrent - 1)])

    def SMRB3D(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx):
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        B, C, T, H, W = sens_maps_updated.shape
        sens_maps_updated_ = sens_maps_updated.reshape(B * C, 1, T, H, W)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)
        sens_maps_updated_ = self.ConvBlockSM3D[idx](sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_)
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, T, H, W)
        sens_maps_updated = sens_maps_updated - self.stepsize * (2 * utils.ifft2(mask * (mask * Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + self.scale * sens_maps_updated_)
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated, dim=2) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate
        return sens_maps_updated

    def forward3D(self, Ref_Kspace_f, Target_Kspace_u, mask, num_low_frequencies):
        rec = []
        SMs = []
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(Target_Kspace_u)
            gate = torch.ones_like(sens_maps_updated).cuda()
        else:
            sens_maps_updated, gate = self.sens_net3D(Target_Kspace_u, num_low_frequencies)

        if Ref_Kspace_f is not None:
            Ref_img = utils.sens_reduce(Ref_Kspace_f, sens_maps_updated)

        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps_updated)  # initialization of Target image
        SMs.append(sens_maps_updated)
        rec.append(Target_img_f)

        #### DDRB blocks #### 
        for idx, DDRB_ in enumerate(self.recurrent):
            if Ref_Kspace_f is None:
                Ref_img = Target_img_f.clone()
                Ref_Kspace_f = utils.sens_expand(Ref_img, sens_maps_updated)

            #### Update of SM by SMRB ####
            if (self.coils != 1) & (idx != 0):
                sens_maps_updated = self.SMRB3D(Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx - 1)
                SMs.append(sens_maps_updated)
                Ref_img = utils.sens_reduce(Ref_Kspace_f, sens_maps_updated)

            #### Update of MR image by DDRB ####
            Target_img_f = DDRB_(Ref_img, Ref_Kspace_f, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate)

            rec.append(Target_img_f)

        return rec, utils.rss(Target_img_f, dim=2), sens_maps_updated, Target_img_f
