"""
Pytorch Lightning module to handle fastMRI and CMRxRecon data. 
Modified from https://github.com/facebookresearch/fastMRI/blob/master/fastmri/pl_modules/data_module.py
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch

import fastmri
from data_utils.mri_datasets import CombinedCmrxReconSliceDataset2024, CmrxReconSliceDataset2024


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        CmrxReconSliceDataset2024, CombinedCmrxReconSliceDataset2024
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedCmrxReconSliceDataset2024):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))


def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True

    return False

class CmrxReconDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        h5py_folder: str,
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
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            challenge: Name of challenge from ('multicoil', 'singlecoil').
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            sample_rate: Fraction of slices of the training data split to use.
                Can be set to less than 1.0 for rapid prototyping. If not set,
                it defaults to 1.0. To subsample the dataset either set
                sample_rate (sample by slice) or volume_sample_rate (sample by
                volume), but not both.
            val_sample_rate: Same as sample_rate, but for val split.
            volume_sample_rate: Fraction of volumes of the training data split
                to use. Can be set to less than 1.0 for rapid prototyping. If
                not set, it defaults to 1.0. To subsample the dataset either
                set sample_rate (sample by slice) or volume_sample_rate (sample
                by volume), but not both.
            val_volume_sample_rate: Same as volume_sample_rate, but for val
                split.
            train_filter: A callable which takes as input a training example
                metadata, and returns whether it should be part of the training
                dataset.
            val_filter: Same as train_filter, but for val split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError(
                "Can set val_sample_rate or val_volume_sample_rate, but not both."
            )

        # NOTE: Our datapath leads to the the Multicoil directory which contains all our modalities
        self.data_path = data_path
        
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.combine_train_val = combine_train_val
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
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
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter

        # if desired, combine train and val together for the train split
        dataset: Union[CmrxReconSliceDataset2024, CombinedCmrxReconSliceDataset2024]
        if is_train and self.combine_train_val:
            data_partitions = [
                "train",
                "val"
            ]
            data_paths = [self.data_path, self.data_path]
            data_transforms = [data_transform, data_transform]
            challenges = [self.challenge, self.challenge]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedCmrxReconSliceDataset2024(
                roots=data_paths,
                transforms=data_transforms,
                challenges=challenges,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                data_partitions=data_partitions
            )

        else:
            data_path = self.data_path 

            dataset = CmrxReconSliceDataset2024(
                root=data_path,
                challenge=self.challenge,
                data_partition=data_partition,
                transform=data_transform,
                use_dataset_cache=self.use_dataset_cache_file,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                raw_sample_filter=raw_sample_filter,
            )

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            data_paths = [
                self.data_path / "train",# / f"{self.challenge}_train",
                self.data_path / "val", # / f"{self.challenge}_val",
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                # NOTE: Fixed so that val and test use correct sample rates
                sample_rate = self.sample_rate  # if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                data_partition = data_path.relative_to(self.data_path) # separates the data_partition without leading /
                _ = CmrxReconSliceDataset2024(
                    root=data_path,
                    challenge=self.challenge,
                    data_partition=data_partition,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--h5py_folder",
            default=None,
            type=str,
            help="Folder name for converted h5py files",
        )
        parser.add_argument(
            "--challenge",
            choices=("singlecoil", "multicoil"),
            default="multicoil",
            type=str,
            help="Which challenge to preprocess for",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--val_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of slices in the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with volume_sample_rate."
            ),
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (train split only). If not "
                "given all will be used. Cannot set together with sample_rate."
            ),
        )
        parser.add_argument(
            "--val_volume_sample_rate",
            default=None,
            type=float,
            help=(
                "Fraction of volumes of the dataset to use (val split only). If not "
                "given all will be used. Cannot set together with val_sample_rate."
            ),
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            action="store_true",
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
