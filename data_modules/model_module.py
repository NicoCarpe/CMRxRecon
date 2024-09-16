"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Modified by Nicolas Carpenter <ngcarpen@ualberta.ca> 
"""

import pathlib
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric

import fastmri
from fastmri import evaluate
from torch import nn
import torch.nn.functional as F
from model.model import ReconModel
from model.utils import ssimloss, psnr, combined_loss, sens_expand, rss, fft2, ifft2


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
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

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


    def forward(self, masked_kspace, mask, num_low_frequencies, max_img_size, attrs):
        return self.recon_model(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)
    

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        # Forward pass
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)

        # Compute combined loss
        loss_all = combined_loss(
            rec_img=rss(rec_img),
            target_img=rss(ifft2(target_k)),
            sens_maps=sens_maps,
            masked_kspace=masked_kspace,
            mask=mask,
            max_val=attrs['max'],
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            lambda_SM=self.hparams.lambda_SM,
        )

        # Log loss every 10 batches
        if batch_idx % 10 == 0:
            self.log_dict({
                "train_loss": loss_all,
            }, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

        # Log the last image of the epoch
        # self.trainer.num_training_batches is a list containing all the batches for each dataloader, so we must sum over it
        if batch_idx == (sum(self.trainer.num_training_batches) - 1):   
            # Convert image tensor to a form suitable for logging
            img_to_log = rss(rec_img)[0].unsqueeze(0)  # Log the first image of the last batch
            self.log_image("train_rec_img_last", img_to_log)

        return loss_all




    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        # Forward pass
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)

        # Compute combined loss
        val_loss = combined_loss(
            rec_img=rss(rec_img),
            target_img=rss(ifft2(target_k)),
            sens_maps=sens_maps,
            masked_kspace=masked_kspace,
            mask=mask,
            max_val=attrs['max'],
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            lambda_SM=self.hparams.lambda_SM,
        )

        # Log loss every 10 batches
        if batch_idx % 10 == 0:
            self.log_dict({
                "val_loss": val_loss,
            }, sync_dist=True, prog_bar=True, batch_size=1)

        # Log the last image of the epoch
        # self.trainer.num_val_batches is a list containing all the batches for each dataloader, so we must sum over it
        if batch_idx == (sum(self.trainer.num_val_batches) - 1):   
            img_to_log = rss(rec_img)[0].unsqueeze(0)  # Log the first image of the last batch
            self.log_image("val_rec_img_last", img_to_log)

        return val_loss




    def on_validation_epoch_end(self):
        # Initialize metric accumulators
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        num_examples = 0

        # Aggregate metrics across all batches
        for val_output in self.validation_outputs:
            num_examples += 1

            # Calculate NMSE, PSNR, SSIM
            nmse = val_output["mse_val"] / val_output["target_norm"]
            psnr = psnr(val_output["mse_val"], val_output["max_val"])
            ssim = val_output["ssim_val"]

            # Accumulate metrics
            metrics["nmse"] += nmse
            metrics["psnr"] += psnr.item()
            metrics["ssim"] += ssim

        # Check if we're doing a sanity check
        if not self.trainer.sanity_checking:
            # Average metrics over all examples
            metrics["nmse"] /= num_examples
            metrics["psnr"] /= num_examples
            metrics["ssim"] /= num_examples

            # Log the final validation metrics
            self.log("val_metrics/nmse", metrics["nmse"], prog_bar=True, sync_dist=True)
            self.log("val_metrics/psnr", metrics["psnr"], prog_bar=True, sync_dist=True)
            self.log("val_metrics/ssim", metrics["ssim"], prog_bar=True, sync_dist=True)

        # Clear the outputs after the epoch
        self.validation_outputs.clear()



    def test_step(self, batch, batch_idx):
        masked_kspace, mask, target_k, attrs, max_img_size = batch
        num_low_frequencies = attrs['num_low_frequencies']

        target = ifft2(target_k)
        target_rss = rss(target)

        # Forward pass
        rec_img, sens_maps = self(masked_kspace, mask, num_low_frequencies, max_img_size, attrs)

        # Compute combined loss
        test_loss = combined_loss(
            rec_img=rss(rec_img),
            target_img=target_rss,
            sens_maps=sens_maps,
            masked_kspace=masked_kspace,
            mask=mask,
            max_val=attrs['max'],
            alpha=self.hparams.alpha,
            beta=self.hparams.beta,
            lambda_SM=self.hparams.lambda_SM,
        )

        # Log every 10 test batches
        if batch_idx % 10 == 0:
            self.log("test_loss", test_loss.item(), prog_bar=True, batch_size=1)

        # Log the last image of the epoch
        # # self.trainer.num_training_batches is a list containing all the batches for each dataloader, so we must sum over it
        if batch_idx == (sum(self.trainer.num_test_batches) - 1):   
            img_to_log = rss(rec_img)[0].unsqueeze(0)  # Log the first image of the last batch
            self.log_image("test_rec_img_last", img_to_log)

        return {
            "test_loss": test_loss,
            "fname": attrs['fname'],
            "slice_ind": attrs['slice_ind'],
            "output": rss(rec_img),
            "target": target,
        }


    def log_image(self, name, image):
        # Log image to TensorBoard
        self.logger.experiment.add_image(name, image, global_step=self.global_step)


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
