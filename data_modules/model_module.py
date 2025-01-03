"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified by Nicolas Carpenter <ngcarpen@ualberta.ca> 
"""

import os
import pathlib
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric
import pytorch_lightning as pl

from fastmri import rss, complex_abs, ifft2c, evaluate, save_reconstructions

from model.model import ReconModel
from model.utils import ssimloss, normalize_kspace, sens_expand, crop_submission  


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity
    

class ReconMRIModule(pl.LightningModule):
    def __init__(self, num_log_images: int = 2, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Save all the provided kwargs into hparams

        # Directly initialize ReconModel with the provided configuration
        self.recon_model = ReconModel(
            coils=kwargs.get('coils'),
            num_heads=kwargs.get('num_heads'),
            window_size=kwargs.get('window_size'),
            depths=kwargs.get('depths'),
            embed_dim=kwargs.get('embed_dim'),
            num_recurrent=kwargs.get('num_recurrent'),
            sens_chans=kwargs.get('sens_chans'),
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def forward(self, masked_kspace, mask, num_low_frequencies, attrs, max_img_size):
        return self.recon_model(masked_kspace, mask, num_low_frequencies, attrs, max_img_size)
    
    
    #########################################################################################################
    # Train Step
    #########################################################################################################
    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        # Forward pass (B, C, T, H, W, comp)
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, attrs, max_img_size)

        # Predicted k-space
        pred_kspace = sens_expand(rec_img, sens_maps)

        # K-space fidelity loss
        pred_kspace_norm = normalize_kspace(pred_kspace)
        target_k_norm = normalize_kspace(target_k)
        loss_fidelity = F.mse_loss(pred_kspace_norm, target_k_norm)

        # Sensitivity map consistency loss
        loss_sensitivity_consistency = F.l1_loss(mask * pred_kspace, masked_kspace)

        # Image-domain SSIM loss
        target_img = rss(complex_abs(ifft2c(target_k)), dim=1)  # (B, T, H, W)
        rec_img_mag = rss(complex_abs(rec_img), dim=1)          # (B, T, H, W)
        loss_ssim = ssimloss(target_img, rec_img_mag, attrs['max'])

        # Combine losses
        loss = (self.hparams.alpha * loss_ssim +
                self.hparams.beta * loss_fidelity +
                self.hparams.lambda_SM * loss_sensitivity_consistency)

        # Log losses
        loss_dict = {
            "train_loss_ssim": loss_ssim.detach(),
            "train_loss_fidelity": loss_fidelity.detach(),
            "train_loss_sensitivity_consistency": loss_sensitivity_consistency.detach(),
            "train_loss": loss.detach()
        }
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=1, sync_dist=True)

        # Optionally log images every N batches
        if batch_idx % 200 == 0:  # adjust frequency as desired
            # Prepare images for logging
            masked_target = rss(complex_abs(ifft2c(masked_kspace)), dim=1)
            sens_maps_mag = rss(complex_abs(sens_maps), dim=1)
            
            # Log a figure with reconstruction, target, sensitivity maps, and masked input
            self.image_figure(
                rec_img_mag, 
                target_img, 
                sens_maps_mag, 
                masked_target, 
                step=self.global_step, 
                mode="Training"
            )

            # If you also want to log just a single image:
            # Normalize and log one of the training images
            output_vis = rec_img_mag[0].unsqueeze(0)
            self.log_image("train/reconstruction", output_vis)

        return loss


    #########################################################################################################
    # Validation Epoch Start
    #########################################################################################################

    def on_validation_epoch_start(self):
        """
        Initialize the list that will collect validation results at the start of each validation epoch.
        """
        super().on_validation_epoch_start()
        self.val_output_list = []  # Reset for the new validation epoch


    #########################################################################################################
    # Validation Step
    #########################################################################################################

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        # Forward pass: (B, C, T, H, W, comp)
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, attrs, max_img_size)

        # Compute predicted k-space
        pred_kspace = sens_expand(rec_img, sens_maps)  # (B, C, T, H, W, 2)

        # Normalize k-space and compute fidelity loss
        pred_kspace_norm = normalize_kspace(pred_kspace)
        target_k_norm = normalize_kspace(target_k)
        loss_fidelity = F.mse_loss(pred_kspace_norm, target_k_norm)

        # Sensitivity map consistency loss
        loss_sensitivity_consistency = F.l1_loss(mask * pred_kspace, masked_kspace)

        # Image-domain SSIM loss
        target_img = rss(complex_abs(ifft2c(target_k)), dim=1)  # (B, T, H, W)
        rec_img_mag = rss(complex_abs(rec_img), dim=1)          # (B, T, H, W)
        val_loss_ssim = ssimloss(target_img, rec_img_mag, attrs['max'])

        # Handle any NaNs
        val_loss_ssim = torch.nan_to_num(val_loss_ssim, nan=1.0)
        loss_fidelity = torch.nan_to_num(loss_fidelity, nan=1.0)
        loss_sensitivity_consistency = torch.nan_to_num(loss_sensitivity_consistency, nan=1.0)

        # Combine validation losses
        val_loss = (
            self.hparams.alpha * val_loss_ssim +
            self.hparams.beta * loss_fidelity +
            self.hparams.lambda_SM * loss_sensitivity_consistency
        )

        # Check output and target dimensions
        if rec_img_mag.ndim != 4 or target_img.ndim != 4:
            raise RuntimeError(f"Unexpected output or target size from validation_step: {rec_img_mag.shape}, {target_img.shape}")

        # Determine validation dataset size based on the type of val_dataloaders
        if isinstance(self.trainer.val_dataloaders, list):
            val_dataset_size = len(self.trainer.val_dataloaders[0].dataset)
        else:
            val_dataset_size = len(self.trainer.val_dataloaders.dataset)

        # Initialize val_log_indices if not already set
        if self.val_log_indices is None:
            # Use the validation dataset size to select random indices
            self.val_log_indices = list(
                np.random.permutation(val_dataset_size)[: self.num_log_images]
            )

        # Determine which samples to log
        for i in range(min(self.num_log_images, masked_kspace.size(0))):
            b_idx = self.val_log_indices[i]  # Use pre-selected indices
            key = f"val_images_idx_{b_idx}"
            target = target_img[i].unsqueeze(0)
            output = rec_img_mag[i].unsqueeze(0)
            error = torch.abs(target - output)

            # Normalize for logging
            output = output / (output.max() + 1e-8)
            target = target / (target.max() + 1e-8)
            error = error / (error.max() + 1e-8)

            # Log images
            self.log_image(f"{key}/target", target)
            self.log_image(f"{key}/reconstruction", output)
            self.log_image(f"{key}/error", error)

        # Initialize the metrics using defaultdict(dict)
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        num_frames = target_img.shape[1]  # Temporal dimension (T)

        for i, fname in enumerate(attrs["fname"]):
            slice_num = int(attrs["slice_ind"][i].cpu())
            maxval = attrs["max"][i].cpu().numpy()
            output = rec_img_mag[i].cpu().numpy()
            target = target_img[i].cpu().numpy()

            # Loop over frames to store metrics in the defaultdicts
            for frame in range(num_frames):
                target_slice = target[frame]
                output_slice = output[frame]

                # Add batch dimension
                target_slice = target_slice[None, ...]  # (1, H, W)
                output_slice = output_slice[None, ...]  # (1, H, W)

                # Compute metrics and store them in the defaultdict
                mse_vals[fname][frame] = torch.tensor(evaluate.mse(target_slice, output_slice)).view(1)
                target_norms[fname][frame] = torch.tensor(evaluate.mse(target_slice, np.zeros_like(target_slice))).view(1)
                ssim_vals[fname][frame] = torch.tensor(evaluate.ssim(target_slice, output_slice, maxval=maxval)).view(1)
            max_vals[fname] = maxval

        # Prepare the result dictionary with metrics by frame
        results = {
            "batch_idx": batch_idx,
            "fname": attrs["fname"],
            "slice_num": attrs["slice_ind"],
            "max_value": attrs["max"],
            "output": rec_img_mag,
            "target": target_img,
            "val_loss": val_loss,
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

        # Accumulate the result into the list
        self.val_output_list.append(results)

        # Determine total validation batches
        if isinstance(self.trainer.num_val_batches, list):
            total_val_batches = sum(self.trainer.num_val_batches)
        else:
            total_val_batches = self.trainer.num_val_batches

        # Make figures every 200 batches if desired
        if (batch_idx % 200 == 0):
            masked_target = rss(complex_abs(ifft2c(masked_kspace)), dim=1)
            sens_maps = rss(complex_abs(sens_maps), dim=1)

            self.image_figure(
                rec_img_mag, 
                target_img, 
                sens_maps,
                masked_target, 
                step=self.global_step, 
                mode="Validation"
            )

        # Return the loss for logging
        return val_loss



    #########################################################################################################
    # Validation Epoch End
    #########################################################################################################

    def on_validation_epoch_end(self):
        val_logs = self.val_output_list  # This contains all results across the validation epoch
    
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        
        # Log final validation loss
        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)

        # Log final metrics to TensorBoard
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

        # Optionally print them to console here
        self.print(
            f"Epoch {self.current_epoch} -- "
            f"Val NMSE: {(metrics['nmse']/tot_examples).item():.6f}, "
            f"Val SSIM: {(metrics['ssim']/tot_examples).item():.6f}, "
            f"Val PSNR: {(metrics['psnr']/tot_examples).item():.2f}"
        )

    #########################################################################################################
    # Test Epoch End
    #########################################################################################################

    def on_test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        save_reconstructions(outputs, save_path)

    
    #########################################################################################################
    # Image Logging
    #########################################################################################################

    def log_image(self, name, image):
        # Log the first slice of the first image in the batch for now
        # Shape (B, T, H, W) -> (1, H, W) for a single image
        image_to_log = image[0, 0, :, :]  # Selecting the first batch and the first temporal slice
        
        # Add a channel dimension for TensorBoard compatibility, resulting in (1, H, W)
        image_to_log = image_to_log.unsqueeze(0)
        
        # Log the image to TensorBoard
        self.logger.experiment.add_image(name, image_to_log, global_step=self.global_step)

    def image_figure(self, rec_img, target_img, sens_maps, masked_target, step, mode="Training"):
        """
        Logs the reconstructed image, target image, and sensitivity maps.

        Parameters:
        - rec_img: Reconstructed image tensor (after applying rss)
        - target_img: Target image tensor (after applying rss and ifft2)
        - sens_maps: Sensitivity maps tensor
        - step: Current step (for logging purposes)
        - mode: A string specifying if it's "Training" or "Validation" (default is "Training")
        """
        rec_img_to_log = rec_img[0, 0, :, :].float().cpu()
        target_img_to_log = target_img[0, 0, :, :].float().cpu()
        sens_maps_to_log = sens_maps[0, 0, :, :].float().cpu()
        masked_target_to_log = masked_target[0, 0, :, :].float().cpu()

        # # Normalize images
        rec_img_to_log = (rec_img_to_log - rec_img_to_log.min()) / (rec_img_to_log.max() - rec_img_to_log.min() + 1e-8)
        target_img_to_log = (target_img_to_log - target_img_to_log.min()) / (target_img_to_log.max() - target_img_to_log.min() + 1e-8)
        sens_maps_to_log = (sens_maps_to_log - sens_maps_to_log.min()) / (sens_maps_to_log.max() - sens_maps_to_log.min() + 1e-8)
        masked_target_to_log = (masked_target_to_log - masked_target_to_log.min()) / (masked_target_to_log.max() - masked_target_to_log.min() + 1e-8)
        
        # Convert to numpy
        rec_img_np = rec_img_to_log.detach().numpy()
        target_img_np = target_img_to_log.detach().numpy()
        sens_maps_np = sens_maps_to_log.detach().numpy()
        masked_target_np = masked_target_to_log.detach().numpy()

        # Create figure with vertically stacked images
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 3 rows, 1 column
        axes[0, 0].imshow(rec_img_np, cmap='gray')
        axes[0, 0].set_title(f'{mode} Reconstructed Image')
        axes[0, 0].axis('off')
        axes[1, 0].imshow(target_img_np, cmap='gray')
        axes[1, 0].set_title(f'{mode} Target Image')
        axes[1, 0].axis('off')
        axes[0, 1].imshow(sens_maps_np, cmap='gray')
        axes[0, 1].set_title(f'{mode} Sensitivity Maps')
        axes[0, 1].axis('off')
        axes[1, 1].imshow(masked_target_np, cmap='gray')
        axes[1, 1].set_title(f'{mode} Masked Image')
        axes[1, 1].axis('off')
        plt.tight_layout()

        # Log the figure directly to TensorBoard
        self.logger.experiment.add_figure(f"{mode} Comparison", fig, global_step=step)
        plt.close(fig)

        # Save the figure to a file (for debugging)
        output_dir = os.path.join("/home/nicocarp/scratch/CMRxRecon/outputs", mode.lower())
        os.makedirs(output_dir, exist_ok=True)
        file_name = f'{mode.lower()}_img_step_{step}.png'
        file_path = os.path.join(output_dir, file_name)
        fig.savefig(file_path)

        
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
        # look to see if this is necessary
        # # Mixed precision support
        # if self.hparams.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()
        #     return {"optimizer": optimizer, "lr_scheduler": scheduler, "scaler": scaler}
        
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
        parser.add_argument('--sens_chans', type=int, default=32, help='Number of channels in sensitivity network')
        parser.add_argument('--alpha', type=float, default=1.0, help='Weight for SSIM loss')
        parser.add_argument('--beta', type=float, default=1.0, help='Weight for Fidelity loss')
        parser.add_argument('--lambda_SM', type=float, default=0.2, help='Weight for Sensitivity Consistency loss')

        parser.add_argument('--GT', type=bool, default=True, help='If there is GT, default is True')

        parser.add_argument('--num_heads', type=int, nargs='+', default=[3, 6, 12, 24], help='Number of attention heads for each transformer stage')
        parser.add_argument('--window_size', type=int, nargs=3, default=(8, 32, 64), help='Window size for each dimension (T, H, W)')
        parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2], help='Depths of each Swin Transformer stage')
        parser.add_argument('--patch_size', type=int, nargs=3, default=(4, 4, 4), help='Patch size for each dimension (T, H, W)')
        parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
        parser.add_argument('--mlp_ratio', type=float, default=4., help='Ratio of mlp hidden dim to embedding dim')
        parser.add_argument('--min_windows', type=int, default=4, help='Minimum number of windows needed for regularization of data')

        return parser
