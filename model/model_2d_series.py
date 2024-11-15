# -*- coding: utf-8 -*-
"""
@author: sunkg

Jul 23, 2024
Modified by Nicolas Carpenter <ngcarpen@ualberta.ca>
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from fastmri import ifft2c, rss

from model import SwinUNet3D, SM_models2D_series, utils


class IRB(nn.Module):
    def __init__(self, coils_all, num_heads, window_size, depths, embed_dim, drop=0., scale=0.1):
        super().__init__()

        self.scale = scale

        # Initialize SwinTransformer3D for image processing
        self.SwinTransformer3D = SwinUNet3D.SwinUnet3D(
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

    def forward(self, target_kspace_u, target_img_f, mask, sens_maps_updated, idx, gate, stepsize):
        target_img_f.requires_grad_(True)

        # Convert to channel dimension representation
        target_img_f = utils.complex_to_chan_dim(target_img_f)

        # SwinTransformer3D forward pass
        output_SwinTransformer3D = self.SwinTransformer3D(target_img_f)

        # Convert back to complex values
        output_SwinTransformer3D = utils.chan_complex_to_last_dim(output_SwinTransformer3D)

        target_img_f = utils.chan_complex_to_last_dim(target_img_f)

        # Gradient update of the image
        target_img_f = target_img_f - stepsize * (2 * utils.sens_reduce(mask * (mask * utils.sens_expand(target_img_f, sens_maps_updated) - target_kspace_u), sens_maps_updated) + self.scale * output_SwinTransformer3D)  

        return target_img_f

class ReconModel(nn.Module): 
    def __init__(self, coils, num_heads, window_size, depths, embed_dim,
                 drop=0., num_recurrent=5, sens_chans=32, 
                 mask_center=True, scale=0.1):
        super().__init__()

        # Initialize sensitivity map network
        self.SMEB = SM_models2D_series.SMEB(
            in_chans=2,
            out_chans=2,
            chans=sens_chans,
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
        self.ConvBlockSM = nn.ModuleList([SM_models2D_series.ConvBlockSM(in_chans=2, out_chans=2) for _ in range(num_recurrent - 1)])

    def SMRB(self, Target_img_f, sens_maps_updated, Target_Kspace_u, mask, gate, idx, stepsize):
        """
        Sensitivity map refinement block (SMRB) with sequential processing over the temporal dimension.
        """
        sens_maps_updated.requires_grad_(True)

        # Expand target image to frequency domain
        Target_Kspace_f = utils.sens_expand(Target_img_f, sens_maps_updated)
        b, c, t, h, w, comp = sens_maps_updated.shape

        # Initialize an empty tensor to store the updated sensitivity maps
        updated_sens_maps = torch.zeros_like(sens_maps_updated)

        # Reshape to batch the coil dimension
        sens_maps_batched = sens_maps_updated.view(b * c, 1, t, h, w, comp)

        # Loop over the temporal dimension
        for time_idx in range(t):
            # Select the temporal slice for processing
            sens_maps_slice = sens_maps_batched[:, :, time_idx, :, :, :]

            # Convert to channel format for ConvBlockSM
            sens_maps_slice = utils.complex_to_chan_dim(sens_maps_slice)

            # Apply ConvBlockSM to the current temporal slice
            sens_maps_slice = self.ConvBlockSM[idx](sens_maps_slice)

            # Convert back to complex format
            sens_maps_slice = utils.chan_complex_to_last_dim(sens_maps_slice)

            # Update the corresponding temporal slice in the final tensor
            updated_sens_maps[:, :, time_idx, :, :, :] = sens_maps_slice

        # Reshape back to original dimensions
        sens_maps_updated = updated_sens_maps.view(b, c, t, h, w, comp)

        # Sensitivity map update
        sens_maps_updated = sens_maps_updated - stepsize * (
            2 * ifft2c(mask * (mask * Target_Kspace_f - Target_Kspace_u) * Target_img_f.conj()) + 
            self.scale * sens_maps_updated
        )
        sens_maps_updated = sens_maps_updated / (rss(sens_maps_updated, dim=1) + 1e-12)
        sens_maps_updated = sens_maps_updated * gate

        return sens_maps_updated



    def forward(self, target_kspace_u, mask, num_low_frequencies, attrs, max_img_size):
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

        # # NOTE: we applied model checkpointing to each of the networks in the IRB, SMEB, and SMRB
        
        # Initialize sensitivity maps with checkpointing
        if self.coils == 1:
            sens_maps_updated = torch.ones_like(target_kspace_u)
            gate = torch.ones_like(sens_maps_updated).cuda()
        else:
            sens_maps_updated, gate = cp.checkpoint(self.SMEB, target_kspace_u, mask, num_low_frequencies, max_img_size, use_reentrant=False)

        sens_maps_updated.requires_grad_(True)

        # Pad the input data to the maximum image size
        target_kspace_u, orginial_size = utils.pad_to_max_size(target_kspace_u, max_img_size)
        mask, _ = utils.pad_to_max_size(mask, max_img_size)

        # Initialize target image
        target_img_f = utils.sens_reduce(target_kspace_u, sens_maps_updated)
        target_img_f.requires_grad_(True)

        # # Recurrent blocks for sensitivity map and MR image update
        # for idx, IRB_ in enumerate(self.recurrent):
        #     if (self.coils != 1) & (idx != 0):
        #         # Checkpoint the SMRB forward pass to reduce memory usage
        #         sens_maps_updated = cp.checkpoint(self.SMRB, target_img_f, sens_maps_updated, target_kspace_u, mask, gate, idx - 1, self.stepsize, use_reentrant=False)

        #     # Checkpoint the IRB forward pass to reduce memory usage
        #     target_img_f = cp.checkpoint(IRB_, target_kspace_u, target_img_f, mask, sens_maps_updated, idx, gate, self.stepsize, use_reentrant=False)
        #     # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        #     # print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

        # Unpad the output data to the original size
        target_img_f = utils.unpad_from_max_size(target_img_f, orginial_size)
        sens_maps_updated = utils.unpad_from_max_size(sens_maps_updated, orginial_size)

        return target_img_f, sens_maps_updated