# -*- coding: utf-8 -*-
"""
@author: sunkg

Jul 23, 2024
Modified by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""
import time
import torch
import torch.nn as nn
from model import SwinUNet3D_no_cyclic, SM_models, utils
from einops import rearrange
import torch.utils.checkpoint as cp


class IRB(nn.Module):
    def __init__(self, coils_all, num_heads, window_size, depths, embed_dim, drop=0., scale=0.1):
        super().__init__()

        
        self.LeakyReLU = nn.LeakyReLU()
        self.coils_all = coils_all
        self.scale = scale

        # Initialize SwinTransformer3D for image processing
        self.SwinTransformer3D = SwinUNet3D_no_cyclic.SwinUnet3D(
            hidden_dim=embed_dim,
            layers=depths,
            heads=num_heads,
            in_channel=2,  # Assuming your input has 2 channels (e.g., real and imaginary)
            num_classes=2,  # Assuming 2 output channels/classes
            head_dim=32,
            window_size=window_size, 
            downscaling_factors=(2, 2, 2, 2),
            relative_pos_embedding=True,
            dropout=drop,
            skip_style='stack',
            stl_channels=32
        )

    def forward(self, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate, stepsize):
        Target_img_f.requires_grad_(True)

        # Convert to channel dimension representation
        Target_img_f = utils.complex_to_chan_dim(Target_img_f)

        # SwinTransformer3D forward pass
        output_SwinTransformer3D = self.SwinTransformer3D(Target_img_f)

        # Convert back to complex values
        output_SwinTransformer3D = utils.chan_dim_to_complex(output_SwinTransformer3D)

        Target_img_f = utils.chan_dim_to_complex(Target_img_f)

        # Gradient update of the image
        Target_img_f = Target_img_f - stepsize * (2 * utils.sens_reduce(mask * (mask * utils.sens_expand(Target_img_f, sens_maps_updated) - Target_Kspace_u), sens_maps_updated) + self.scale * output_SwinTransformer3D)  

        return Target_img_f

class ReconModel(nn.Module): 
    def __init__(self, coils, num_heads, window_size, depths, embed_dim,
                 drop=0., num_recurrent=5, sens_chans=8, sens_steps=4, 
                 mask_center=True, scale=0.1):
        super().__init__()

        # Initialize sensitivity map network
        self.SMEB3D = SM_models.SMEB3D(
            chans=sens_chans,
            sens_steps=sens_steps,
            mask_center=mask_center
        )

        self.scale = scale
        self.coils = coils
        self.stepsize = nn.Parameter(0.1 * torch.rand(1))
        self.num_recurrent = num_recurrent
        # Initialize recurrent blocks (DDRB)
        self.recurrent = nn.ModuleList([IRB(coils_all=coils,
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             depths=depths, 
                                             embed_dim=embed_dim, 
                                             drop=drop,
                                             scale=scale,
                                            ) 
                                        for _ in range(num_recurrent)])

        # Initialize convolution blocks for sensitivity map refinement
        self.ConvBlockSM3D = nn.ModuleList([SM_models.ConvBlockSM2D_t(in_chans=2, conv_num=2) for _ in range(num_recurrent - 1)])

    def SMRB3D(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx, stepsize):
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
        sens_maps_updated.requires_grad_(True)

        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        B, C, T, X, Y = sens_maps_updated.shape

        # Reshape and apply convolution
        sens_maps_updated_ = sens_maps_updated.reshape(B * C, 1, T, X, Y)
        sens_maps_updated_ = utils.complex_to_chan_dim(sens_maps_updated_)
        # use checkpointing for heavy computation 
        sens_maps_updated_ = self.ConvBlockSM3D[idx](sens_maps_updated_)
        sens_maps_updated_ = utils.chan_dim_to_complex(sens_maps_updated_)
        sens_maps_updated_ = sens_maps_updated_.reshape(B, C, T, X, Y)

        # Sensitivity map update
        sens_maps_updated = sens_maps_updated - stepsize * (
            2 * utils.ifft2(mask * (mask * Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + 
            self.scale * sens_maps_updated_
        )
        sens_maps_updated = sens_maps_updated / (utils.rss(sens_maps_updated) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate

        return sens_maps_updated

    def forward(self, Target_Kspace_u, mask, num_low_frequencies, max_img_size, attrs):
        """
        Forward pass for ReconModel.
        Args:
            Target_Kspace_u: Undersampled K-space data.
            mask: K-space mask.
            num_low_frequencies: Number of low-frequency components.
            max_img_size: Maximum image size for padding.
        Returns:
            RSS images, sensitivity maps, and final image.
        """

        # NOTE: we applied model checkpointing to each of the networks in the IRB, SMEB, and SMRB
        
        # Initialize sensitivity maps with checkpointing
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(Target_Kspace_u)
            gate = torch.ones_like(sens_maps_updated).cuda()
        else:
            sens_maps_updated, gate = cp.checkpoint(self.SMEB3D, Target_Kspace_u, num_low_frequencies, max_img_size, use_reentrant=False)

        sens_maps_updated.requires_grad_(True)

        # Pad the input data to the maximum image size
        Target_Kspace_u, orginial_size = utils.pad_to_max_size(Target_Kspace_u, max_img_size)
        mask, _ = utils.pad_to_max_size(mask, max_img_size)

        # Initialize target image
        Target_img_f = utils.sens_reduce(Target_Kspace_u, sens_maps_updated)
        Target_img_f.requires_grad_(True)

        # Recurrent blocks for sensitivity map and MR image update
        for idx, IRB_ in enumerate(self.recurrent):
            if (self.coils != 1) & (idx != 0):
                # Checkpoint the SMRB forward pass to reduce memory usage
                sens_maps_updated = cp.checkpoint(self.SMRB3D, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx - 1, self.stepsize, use_reentrant=False)

                # no checkpoint test
                # sens_maps_updated = self.SMRB3D(Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx - 1, self.stepsize)

            # Checkpoint the IRB forward pass to reduce memory usage
            Target_img_f = cp.checkpoint(IRB_, Target_Kspace_u, Target_img_f, mask, sens_maps_updated, idx, gate, self.stepsize, use_reentrant=False)
            
        # Unpad the output data to the original size
        Target_img_f = utils.unpad_from_max_size(Target_img_f, orginial_size)
        sens_maps_updated = utils.unpad_from_max_size(sens_maps_updated, orginial_size)

        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)


        return Target_img_f, sens_maps_updated
