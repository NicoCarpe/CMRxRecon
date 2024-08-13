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
    def __init__(self, coils_all, num_heads, window_size, depths, embed_dim, 
                 patch_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, ds_ref=True, scale=0.1):
        super().__init__()

        self.stepsize = nn.Parameter(0.1 * torch.rand(1))
        self.LeakyReLU = nn.LeakyReLU()
        self.coils_all = coils_all
        self.ds_ref = ds_ref
        self.scale = scale
        
        # Initialize CNN for K-space processing
        self.CNN = CNN.NormNet3D(in_chans=coils_all * 2, out_chans=coils_all, chans=32)  

        # Initialize SwinTransformer3D for image processing
        self.SwinTransformer3D = SwinTransformer3D.SwinTransformer3D(
            patch_size=patch_size,
            in_chans=2,  # assuming 2 channels for real and imaginary values
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
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

    def forward(self, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate, max_img_size):
        original_size = Target_img_f.shape[1:]

        # Pad the inputs to the maximum size
        Target_img_f_padded, pad_params_img = utils.pad_to_max_size(Target_img_f, max_img_size)
        Target_Kspace_u_padded, pad_params_kspace = utils.pad_to_max_size(Target_Kspace_u, max_img_size)
        padding_mask = utils.create_padding_mask(Target_img_f, max_img_size)

        # Convert to channel dimension representation
        Target_Kspace_f = utils.sens_expand(Target_img_f_padded, sens_maps_updated)
        Target_Kspace_f = utils.complex_to_chan_dim(Target_Kspace_f)
        Target_img_f_padded = utils.complex_to_chan_dim(Target_img_f_padded)

        # CNN and SwinTransformer3D forward pass
        output_CNN = self.CNN(Target_Kspace_f)
        output_SwinTransformer3D = self.SwinTransformer3D(Target_img_f_padded, padding_mask)

        # Convert back to complex values
        output_CNN = utils.chan_dim_to_complex(output_CNN)
        output_SwinTransformer3D = utils.chan_dim_to_complex(output_SwinTransformer3D)
        Target_img_f_padded = utils.chan_dim_to_complex(Target_img_f_padded)

        # Sensitivity map and image refinement
        Target_Kspace_f_down = utils.sens_expand(Target_img_f_padded, sens_maps_updated)
        term1 = 2 * utils.sens_reduce(mask * (mask * Target_Kspace_f_down - Target_Kspace_u_padded), sens_maps_updated)
        term2 = utils.sens_reduce(output_CNN, sens_maps_updated)

        # Update the MR image with scaling
        Target_img_f_padded = Target_img_f_padded - self.stepsize * (term1 + self.scale * term2 + self.scale * output_SwinTransformer3D)

        # Unpad the output to its original size
        Target_img_f = utils.unpad_from_max_size(Target_img_f_padded, original_size)

        return Target_img_f

class fD2RT3D(nn.Module):
    def __init__(self, coils, num_heads, window_size, depths, patch_size, embed_dim,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, n_SC=1, num_recurrent=5, sens_chans=8,
                 sens_steps=4, mask_center=True, ds_ref=True, scale=0.1):
        super().__init__()

        # Initialize sensitivity map network
        self.SMEB3D = sensitivity_model.SMEB3D(
            chans=sens_chans,
            sens_steps=sens_steps,
            mask_center=mask_center
        )
        self.ds_ref = ds_ref
        self.scale = scale
        self.coils = coils
        self.stepsize = nn.Parameter(0.1 * torch.rand(1))
        self.num_recurrent = num_recurrent

        # Initialize recurrent blocks (DDRB)
        self.recurrent = nn.ModuleList([DDRB(coils_all=coils,
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             depths=depths, 
                                             patch_size=patch_size,
                                             embed_dim=embed_dim, 
                                             mlp_ratio=mlp_ratio, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale, 
                                             drop=drop, 
                                             attn_drop=attn_drop, 
                                             drop_path=drop_path,
                                             norm_layer=norm_layer, 
                                             n_SC=n_SC, 
                                             ds_ref=ds_ref, 
                                             scale=scale) 
                                        for _ in range(num_recurrent)])

        # Initialize convolution blocks for sensitivity map refinement
        self.ConvBlockSM3D = nn.ModuleList([CNN.ConvBlockSM3D(in_chans=2, conv_num=2) for _ in range(num_recurrent - 1)])

    def SMRB3D(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx):
        """
        Sensitivity map refinement block (SMRB).
        Args:
            Target_img_f: Target image in frequency domain.
            sens_maps_updated: Updated sensitivity maps.
            Target_Kspace_u: Undersampled K-space data.
            mask: K-space mask.
            gate: Gating parameter.
            idx: Index of the current recurrent block.
        Returns:
            Updated sensitivity maps.
        """
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        B, T, C, H, W = sens_maps_updated.shape
        
        # Reshape and apply convolution
        sens_maps_updated_ = sens_maps_updated.permute(0, 2, 1, 3, 4).reshape(B * C, T, 1, H, W)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)
        sens_maps_updated_ = self.ConvBlockSM3D[idx](sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_)
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
        
        # Sensitivity map update
        sens_maps_updated = sens_maps_updated - self.stepsize * (
            2 * utils.ifft2(mask * (mask * Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + 
            self.scale * sens_maps_updated_
        )
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated, dim=1) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate
        return sens_maps_updated

    def forward(self, Target_Kspace_u, mask, num_low_frequencies, max_img_size):
        """
        Forward pass for fD2RT3D model.
        Args:
            Target_Kspace_u: Undersampled K-space data.
            mask: K-space mask.
            num_low_frequencies: Number of low-frequency components.
            max_img_size: Maximum image size for padding.
        Returns:
            Reconstructed images, RSS images, sensitivity maps, and final image.
        """
        rec = []
        SMs = []

        # Pad the input data to the maximum image size
        Target_Kspace_u, pad_sizes = utils.pad_to_max_size(Target_Kspace_u, max_img_size)
        mask, _ = utils.pad_to_max_size(mask, max_img_size)

        # Initialize sensitivity maps
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(Target_Kspace_u)
            gate = torch.ones_like(sens_maps_updated).cuda()
        else:
            sens_maps_updated, gate = self.SMEB3D(Target_Kspace_u, num_low_frequencies, max_img_size)

        # Initialize target image
        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps_updated)
        SMs.append(sens_maps_updated)
        rec.append(Target_img_f)

        # Recurrent blocks for sensitivity map and MR image update
        for idx, DDRB_ in enumerate(self.recurrent):
            if (self.coils != 1) & (idx != 0):
                sens_maps_updated = self.SMRB3D(Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx - 1)
                SMs.append(sens_maps_updated)

            Target_img_f = DDRB_(Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate, max_img_size)
            rec.append(Target_img_f)

        # Unpad the output data to the original size
        rec = [utils.unpad_from_max_size(r, pad_sizes) for r in rec]
        Target_img_f = utils.unpad_from_max_size(Target_img_f, pad_sizes)
        sens_maps_updated = utils.unpad_from_max_size(sens_maps_updated, pad_sizes)

        return rec, utils.rss(Target_img_f, dim=1), sens_maps_updated, Target_img_f
