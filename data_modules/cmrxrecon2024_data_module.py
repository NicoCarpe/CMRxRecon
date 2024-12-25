"""
Pytorch Lightning module to handle fastMRI and CMRxRecon data. 
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/pl_modules/data_module.py
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
import os

import fastmri
from data_utils.dataset_cmrxrecon2024 import CombinedVolumeDataset, VolumeDataset

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        VolumeDataset, CombinedVolumeDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP (Distributed Data Parallel)
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedVolumeDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if is_ddp:
                    # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    # Non-DDP: Seed is just adjusted by worker id and dataset index
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:
            # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))

def _check_both_not_none(val1, val2):
    """Check if both values are not None. Used to ensure mutually exclusive arguments."""
    if (val1 is not None) and (val2 is not None):
        return True
    return False

class CmrxReconDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI and CMRxRecon data sets.

    This class handles configurations for training and validation data.
    It abstracts away data handling specifics from the main training script.

    Note that subsampling mask and transform configurations are expected to be
    passed into this data module.
    """

    def __init__(
        self,
        data_path: Path,
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        combine_train_val: bool = False,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to the root data directory.
            challenge: Type of challenge, 'multicoil' or 'singlecoil'.
            train_transform: Transform function applied to the training data.
            val_transform: Transform function applied to the validation data.
            combine_train_val: Whether to combine train and val splits into one dataset.
            sample_rate: Fraction of slices of the training data split to use.
            val_sample_rate: Fraction of slices of the validation data split to use.
            volume_sample_rate: Fraction of volumes of the training data split to use.
            val_volume_sample_rate: Fraction of volumes of the validation data split to use.
            train_filter: Optional filter for training data.
            val_filter: Optional filter for validation data.
            use_dataset_cache_file: Whether to cache dataset metadata.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loaders.
            distributed_sampler: Whether to use a distributed sampler (for DDP).
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Cannot set both sample_rate and volume_sample_rate.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Cannot set both val_sample_rate and val_volume_sample_rate.")

        # Store initialization parameters
        self.data_path = data_path
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.combine_train_val = combine_train_val
        self.sample_rate = sample_rate
        self.val_sample_rate = sample_rate              # for now we will just have the sample rates tied to a common input value
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for a given data partition (train/val).

        Args:
            data_transform: Transform function applied to the data.
            data_partition: Which data partition to load ('train', 'val').
            sample_rate: Fraction of slices to load (for sampling).
            volume_sample_rate: Fraction of volumes to load (for volume sampling).

        Returns:
            A PyTorch DataLoader instance.
        """
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            sample_rate = (
                self.val_sample_rate if sample_rate is None else sample_rate
            )
            volume_sample_rate = (
                self.val_volume_sample_rate if volume_sample_rate is None else volume_sample_rate
            )
            raw_sample_filter = self.val_filter

        # Combine train and val data if specified
        if is_train and self.combine_train_val:
            # TODO: This needs to be fixed to account for new train/val splits
            data_partitions = ["train", "val"]
            data_paths = [self.data_path, self.data_path]
            data_transforms = [self.train_transform, self.val_transform]
            challenges = [self.challenge, self.challenge]

            # Prepare sampling rates for combined dataset
            sample_rates, volume_sample_rates = None, None
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]

            dataset = CombinedVolumeDataset(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter
            )
        else:
            dataset = VolumeDataset(
                root=self.data_path,
                challenge=self.challenge,
                transform=self.train_transform if data_partition == "train" else self.val_transform,
                use_dataset_cache=self.use_dataset_cache_file,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                raw_sample_filter=raw_sample_filter,
            )
        

        # Handle data shuffling and distributed sampling
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        
        print(f"Length of {data_partition} dataset: {len(dataset)}")

        # Return the configured DataLoader
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
            pin_memory=True,  # Recommended if you're using a GPU
        )

    def prepare_data(self):
        """
        Prepare data by initializing the dataset to ensure cache setup.
        This method is used to make sure the dataset is cached on rank 0 DDP process.
        """
        if self.use_dataset_cache_file:
            data_partitions = ["train", "val"]

            for data_partition in data_partitions:
                # Use the base path without modification
                _ = VolumeDataset(
                    root=self.data_path,  # Use the base path directly
                    challenge=self.challenge,
                    transform=self.train_transform if data_partition == "train" else self.val_transform,
                    sample_rate=self.sample_rate,
                    volume_sample_rate=self.volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file
                )


    def train_dataloader(self):
        """Return DataLoader for training data."""
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        """Return DataLoader for validation data."""
        return self._create_data_loader(self.val_transform, data_partition="val")

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that are specific to this data module.
        These arguments will be added to the main argument parser.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Dataset-related arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to the root data directory.",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="multicoil",
            type=str,
            help="Which challenge type to preprocess data for.",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices to use for training (between 0 and 1).",
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help="Fraction of slices to use for validation (between 0 and 1).",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes to use for training (between 0 and 1).",
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes to use for validation (between 0 and 1).",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pickle file.",
        )
        parser.add_argument(
            "--combine_train_val",
            action="store_true",
            help="Whether to combine train and validation datasets for training.",
        )

        # DataLoader-related arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Batch size for DataLoader."
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers for DataLoader.",
        )

        return parser
