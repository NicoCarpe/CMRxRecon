import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml

#########################################################################################################
# CMRxRecon 2024 part
#########################################################################################################

class CMRxReconRawDataSample2024(NamedTuple):
    fname: Path
    slice_ind: int
    dataset: str
    mask: Path

class CombinedCmrxReconSliceDataset2024(torch.utils.data.Dataset):\
    """
    A container for combining slice datasets.
    """
    
    def __init__(
        self,
        roots: Sequence[Path],
        challenges: Sequence[str],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        data_partitions: Sequence[str]
    ):
        """
        Args:
            roots: Paths to the datasets.
            challenges: "singlecoil" or "multicoil" depending on which
                challenge to use.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form. The transform
                function should take 'kspace', 'target', 'attributes',
                'filename', and 'slice' as inputs. 'target' may be null for
                test data.
            sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the slices should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            volume_sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the volumes should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
            data_partitions: A sequence of "train", "val", or "test" depending on 
                which part of the dataset is being looked at for each dataset.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )
        if transforms is None:
            transforms = [None] * len(roots)
        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not (
            len(roots)
            == len(transforms)
            == len(challenges)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError(
                "Lengths of roots, transforms, challenges, sample_rates do not match"
            )

        self.datasets = []
        self.raw_samples: List[CMRxReconRawDataSample2024] = []
        for i in range(len(roots)):
            self.datasets.append(
                CmrxReconSliceDataset2024(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                    raw_sample_filter=raw_sample_filter,
                    data_partition=data_partitions[i]
                )
            )

            self.raw_samples = self.raw_samples + self.datasets[-1].raw_samples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class CmrxReconSliceDataset2024(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        data_partition: str
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
            data_partition: "train", "val", or "test" depending on which part of 
                the dataset is being looked at.
            
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}
    
        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            
            data_type_list = ['Aorta', 'Cine', 'Mapping', 'Tagging']

            # iterate though all of the modalities
            for data_type in data_type_list:
                task_dir = os.path.join(root, data_type, 'TrainingSet/', 'h5_FullSample', data_partition)
                
                # iterate through all of the patients
                for ks_dir_info in os.scandir(task_dir):
                    if ks_dir_info.is_dir() and ks_dir_info.name not in ['.', '..', '.DS_Store']:
                        full_ks_dir_path = ks_dir_info.path
                        
                        #iterate through all of the views/weightings/etc...
                        for ks_file_info in os.scandir(full_ks_dir_path):
                            if ks_file_info.is_file() and ks_file_info.name not in ['.', '..', '.DS_Store']:
                                full_ks_file_path = ks_file_info.path
                                with h5py.File(full_ks_file_path, 'r') as hf:
                                    # demensions of the data are (z, t, c, x, y)
                                    num_slices = hf["kspace"].shape[0]

                                    # divide our image into individual slices
                                    for slice_ind in range(num_slices):
                                        
                                        # each slice will have multiple masks, each combination being an indiviual sample
                                        for mask_path in hf['masks']:
                                            self.raw_samples += CMRxReconRawDataSample2024(full_ks_file_path, slice_ind, data_type, mask_path)
                     
            # cache the dataset if it not already cached
            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
    
    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i: int):
        fname, slice_ind, data_type, mask_path = self.raw_samples[i]

        with h5py.File(str(fname),'r') as hf:
            kspace = np.asarray(hf["kspace"][slice_ind])
            target = np.asarray(hf[self.recons_key][slice_ind]) if self.recons_key in hf else None
            attrs = dict(hf.attrs)
        
        with h5py.File(str(mask_path),'r') as hf:
            keys = list(hf.keys())[0]
            # transpose the mask into a fastMRI-like format (t, x, y)
            mask = np.asarray(hf[keys]).traspose(2,0,1)
            
        if self.transform is None:
            sample = (kspace, mask, target, attrs, str(fname))
        else:
            sample = self.transform(kspace, mask, target, attrs, str(fname))

        return sample
    