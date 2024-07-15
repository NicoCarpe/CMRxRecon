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
from project.module.data_utils.mri_data import CombinedCmrxReconDataset2024, CmrxReconDataset2024


def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        CmrxReconDataset2024, CombinedCmrxReconDataset2024
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedCmrxReconDataset2024):
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
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
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
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_split: Name of test split from ("test", "challenge").
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        # TODO: ugly but working code, mapping data will be loaded in CmrxReconDataset2024
        self.data_path = data_path / 'Cine' / 'TrainingSet' / h5py_folder 
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
    ) -> torch.utils.data.DataLoader:
        if data_partition == "train":
            is_train = True
        else:
            is_train = False

        # if desired, combine train and val together for the train split
        dataset: Union[CmrxReconDataset2024, CombinedCmrxReconDataset2024]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]
            data_transforms = [data_transform, data_transform]
           
            dataset = CombinedCmrxReconDataset2024(
                roots=data_paths,
                transforms=data_transforms,
                use_dataset_cache=self.use_dataset_cache_file
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / data_partition #"val" 

            dataset = CmrxReconDataset2024(
                root=data_path,
                transform=data_transform,
                use_dataset_cache=self.use_dataset_cache_file
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
            num_workers= self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    
    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / "val" #None #self.data_path #/ f"{self.challenge}_test"
            data_paths = [
                self.data_path / "train",# / f"{self.challenge}_train",
                self.data_path / "val", # / f"{self.challenge}_val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                _ = CmrxReconDataset2024(
                    root=data_path,
                    transform=data_transform,
                    use_dataset_cache=self.use_dataset_cache_file,
                )

    def train_dataloader(self):
        return self._create_data_loader(self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.val_transform, data_partition="val")

    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform, data_partition=self.test_split
        )

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
            help="folder name for converted h5py files",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--test_split",
            choices=("val", "test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
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
