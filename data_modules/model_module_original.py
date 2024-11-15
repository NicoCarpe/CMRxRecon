"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified by Nicolas Carpenter <ngcarpen@ualberta.ca> 
"""

import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch import view_as_real
from torchmetrics.metric import Metric
import matplotlib.pyplot as plt
from fastmri import rss, complex_abs, ifft2c
import torch.nn.functional as F
from model.model import ReconModel
import model.utils as utils
from model.utils import ssimloss, nmse, combined_loss


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class ReconMRIModule(pl.LightningModule):
    def __init__(self, num_log_images=16, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Save all the provided kwargs into hparams
        
        self.num_log_images = num_log_images
        self.val_log_indices = None

        # Initialize metrics for validation
        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()

        # Directly initialize ReconModel with the provided configuration
        self.recon_model = ReconModel(
            coils=kwargs.get('coils'),
            num_heads=kwargs.get('num_heads'),
            window_size=kwargs.get('window_size'),
            depths=kwargs.get('depths'),
            embed_dim=kwargs.get('embed_dim'),
            num_recurrent=kwargs.get('num_recurrent'),
            sens_chans=kwargs.get('sens_chans'),
            sens_steps=kwargs.get('sens_steps'),
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize storage for validation outputs
        self.validation_outputs = []
        self.training_outputs = []


    def forward(self, masked_kspace, mask, num_low_frequencies, attrs, max_img_size):
        return self.recon_model(masked_kspace, mask, num_low_frequencies, attrs, max_img_size)
    
    
    #########################################################################################################
    # Train Step
    #########################################################################################################

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        target_k = target_k.to(device)

        # Forward pass
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)
        

        # NOTE: the rec_img has a its coil dimension preserved, so we just want to drop that dimension here
        rec_img = torch.squeeze(complex_abs(rec_img), dim=1) 
        target_img = rss(complex_abs(ifft2c(target_k)), dim=1)

        max_val=attrs['max']
        print(f"max_val: {max_val}", flush=True)
        # Compute combined loss
        loss_all = combined_loss(
            rec_img=rec_img,
            target_img=target_img,
            sens_maps=sens_maps,
            masked_kspace=masked_kspace,
            mask=mask,
            max_val=attrs['max'],
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            lambda_SM=self.hparams.lambda_SM,
        )

        # Log loss
        self.log_dict({
            "train_loss": loss_all,
        }, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        # Determine total training batches
        if isinstance(self.trainer.num_training_batches, list):
            total_train_batches = sum(self.trainer.num_training_batches)
        else:
            total_train_batches = self.trainer.num_training_batches

        # Log the last image of the epoch
        if batch_idx == (total_train_batches - 1):
            print(f"kspace :min {target_k.min()}, max {target_k.max()}", flush=True)
            self.log_images(
                rec_img, 
                target_img, 
                step=self.global_step, 
                mode="Training"
            )

        return loss_all


    #########################################################################################################
    # Validation Step
    #########################################################################################################

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        target_k = target_k.to(device)

        # Forward pass
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)

        # NOTE: the rec_img has a its coil dimension preserved, so we just want to drop that dimension here
        rec_img = torch.squeeze(complex_abs(rec_img), dim=1) 
        target_img = rss(complex_abs(ifft2c(target_k)), dim=1)

        # Compute metrics
        max_val = attrs['max']
        nmse_val = nmse(rec_img, target_img)
        ssim_val = 1 - ssimloss(rec_img, target_img, max_val) # Recover the SSIM value from the SSIM loss
        psnr = utils.psnr(rec_img, target_img, max_val)
        
        # Accumulate outputs for metrics computation later
        val_output = {
            "max_val": max_val.item(),
            "nmse_val": nmse_val.item(),
            "ssim_val": ssim_val.item(),
            "psnr": psnr.item()
        }

        self.validation_outputs.append(val_output)  # Append to list for later aggregation

        # Compute combined loss
        val_loss = combined_loss(
            rec_img=rec_img,
            target_img=target_img,
            sens_maps=sens_maps,
            masked_kspace=masked_kspace,
            mask=mask,
            max_val=max_val,
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            lambda_SM=self.hparams.lambda_SM,
        )

        # Log loss
        self.log_dict({
            "validation_loss": val_loss,
            "nmse_val": nmse_val,
            "ssim_val": ssim_val,
            "psnr": psnr,
        }, sync_dist=True, prog_bar=True, batch_size=1)

        # Determine total validation batches
        if isinstance(self.trainer.num_val_batches, list):
            total_val_batches = sum(self.trainer.num_val_batches)
        else:
            total_val_batches = self.trainer.num_val_batches

        # Log the last image of the epoch
        if batch_idx == (total_val_batches - 1):
            self.log_images(
                rec_img, 
                target_img, 
                step=self.global_step, 
                mode="Validation"
            )

        return val_loss


    #########################################################################################################
    # Validation Epoch End
    #########################################################################################################

    def on_validation_epoch_end(self):
        # Initialize metric accumulators
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        num_examples = 0


        # Aggregate metrics across all batches
        for val_output in self.validation_outputs:
            num_examples += 1

            # Calculate NMSE, PSNR, SSIM
            nmse = val_output["nmse_val"]
            ssim = val_output["ssim_val"]
            psnr = val_output["psnr"]

            # Accumulate metrics
            metrics["nmse"] += nmse
            metrics["ssim"] += ssim
            metrics["psnr"] += psnr

        # Check if we're doing a sanity check
        if not self.trainer.sanity_checking:
            # Average metrics over all examples
            metrics["nmse"] /= num_examples
            metrics["ssim"] /= num_examples
            metrics["psnr"] /= num_examples

            # Log the final validation metrics
            self.log("val_metrics/nmse", metrics["nmse"], prog_bar=True, sync_dist=True)
            self.log("val_metrics/ssim", metrics["ssim"], prog_bar=True, sync_dist=True)
            self.log("val_metrics/psnr", metrics["psnr"], prog_bar=True, sync_dist=True)
    
    
    #########################################################################################################
    # Image Logging
    #########################################################################################################

    def log_images(self, rec_img, target_img, step, mode="Training"):
        """
        Logs the reconstructed image, target image, and k-space image.

        Parameters:
        - rec_img: Reconstructed image tensor (after applying rss)
        - target_img: Target image tensor (after applying rss and ifft2)
        - target_k: k-space data tensor
        - step: Current step (for logging purposes)
        - mode: A string specifying if it's "Training" or "Validation" (default is "Training")
        """
        rec_img_to_log = rec_img[0, 0, :, :].float().cpu()
        target_img_to_log = target_img[0, 0, :, :].float().cpu()

        # Normalize images
        rec_img_to_log = (rec_img_to_log - rec_img_to_log.min()) / (rec_img_to_log.max() - rec_img_to_log.min() + 1e-8)
        target_img_to_log = (target_img_to_log - target_img_to_log.min()) / (target_img_to_log.max() - target_img_to_log.min() + 1e-8)

        # Convert to numpy
        rec_img_np = rec_img_to_log.detach().numpy()
        target_img_np = target_img_to_log.detach().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rec_img_np, cmap='gray')
        axes[0].set_title(f'{mode} Reconstructed Image')
        axes[0].axis('off')
        axes[1].imshow(target_img_np, cmap='gray')
        axes[1].set_title(f'{mode} Target Image')
        axes[1].axis('off')
        plt.tight_layout()

        # Log the figure directly to TensorBoard
        self.logger.experiment.add_figure(f"{mode} Comparison", fig, global_step=step)
        plt.close(fig)

        # Save the figure to a file (for debugging)
        output_dir = os.path.join("/home/nicocarp/scratch/CMR-Reconstruction/outputs", mode.lower())  # Use os.path.join
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        file_name = f'{mode.lower()}_img_step_{step}.png'  # Create the filename
        file_path = os.path.join(output_dir, file_name)  # Full file path
        fig.savefig(file_path)  # Save the figure


    #########################################################################################################
    # Optimizer Configuration
    #########################################################################################################

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.hparams.lr_step_size, 
            gamma=self.hparams.lr_gamma
        )
        
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", default=0.0003, type=float, help="Adam learning rate")
        parser.add_argument("--lr_step_size", default=40, type=int, help="Epoch at which to decrease step size")
        parser.add_argument("--lr_gamma", default=0.1, type=float, help="Extent to which step size should be decreased")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Strength of weight decay regularization")
        parser.add_argument("--use_checkpoint", action="store_true", help="Use checkpoint (default: False)")

        parser.add_argument('--coils', type=int, default=10, help='Number of coils')
        parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
        parser.add_argument('--num_recurrent', type=int, default=25, help='Number of DCRBs')
        parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels in sensitivity network')
        parser.add_argument('--sens_steps', type=int, default=4, help='Number of steps in initial sensitivity network')
        parser.add_argument('--lambda0', type=float, default=10, help='Weight of the kspace loss')
        parser.add_argument('--lambda1', type=float, default=10, help='Weight of the consistency loss in K-space')
        parser.add_argument('--lambda2', type=float, default=1, help='Weight of the SSIM loss')
        parser.add_argument('--lambda3', type=float, default=1e2, help='Weight of the TV Loss')
        parser.add_argument('--GT', type=bool, default=True, help='If there is GT, default is True')

        parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24], help='Number of attention heads for each transformer stage')
        parser.add_argument('--window_size', type=int, nargs=3, default=(8, 32, 64), help='Window size for each dimension (T, H, W)')
        parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 18, 2], help='Depths of each Swin Transformer stage')
        parser.add_argument('--patch_size', type=int, nargs=3, default=(4, 4, 4), help='Patch size for each dimension (T, H, W)')
        parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
        parser.add_argument('--mlp_ratio', type=float, default=4., help='Ratio of mlp hidden dim to embedding dim')
        parser.add_argument('--min_windows', type=int, default=4, help='Minimum number of windows needed for regularization of data')

        return parser
