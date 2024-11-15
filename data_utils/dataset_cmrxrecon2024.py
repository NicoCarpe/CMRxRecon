"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pickle
import random
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

import h5py
import numpy as np
import torch

import data_utils.utils as utils


class CMRxReconRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    attrs: Dict[str, Any]
    mask_path: Path

class CombinedVolumeDataset(torch.utils.data.Dataset):
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
        dataset_cache_file: Union[str, Path, os.PathLike] = "/home/nicocarp/scratch/CMR-Reconstruction/outputs/dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
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
        self.raw_samples: List[CMRxReconRawDataSample] = []
        for i in range(len(roots)):
            self.datasets.append(
                VolumeDataset(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                    raw_sample_filter=raw_sample_filter,
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


class VolumeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image spatiotemporal volumes.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "/home/nicocarp/scratch/CMR-Reconstruction/outputs/dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
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

        # Keep track of the max image size for regularization of inputs
        # used extra padding for convienient work in the cnns and transformer
        # NOTE: This is probably not the nicest solution, look for better alternatives in the future
        self.max_img_size = [32, 256, 512]  # (t, h, w)

        # Check if dataset is in cache
        if dataset_cache.get(root) is None or not use_dataset_cache:
            
            data_type_list = ['Aorta', 'Cine', 'Mapping', 'Tagging']

            # Iterate through all the modalities
            for data_type in data_type_list:
                task_dir = os.path.join(root, data_type, 'TrainingSet', 'FullSample_test')

                # Iterate through all of the patients
                for patient_dir_info in os.scandir(task_dir):
                    if patient_dir_info.is_dir() and patient_dir_info.name not in ['.', '..', '.DS_Store']:
                        patient_dir = patient_dir_info.path
                            
                        # Iterate through all of the views/weightings/etc. for this patient
                        for ks_file_info in os.scandir(patient_dir):
                            if ks_file_info.is_file() and ks_file_info.name not in ['.', '..', '.DS_Store']:
                                full_ks_file_path = ks_file_info.path  

                                # Open and process the file
                                kdata = utils.load_kdata(full_ks_file_path)

                                # for ease we will rearrange (nt, nz, nc, ny, nx, 2) --> (nc, nt, ny, nx, nz, 2) 
                                kdata = kdata.transpose(2,0,3,4,1,5)

                                # Shape of k-space: [nc, nt, ny, nx, nz, 2]
                                _, _, _, _, num_slices, _  = kdata.shape
                                
                                attrs = {
                                    'max': kdata.max(),
                                    'patient_id': patient_dir[-4:-1],
                                    'shape': kdata.shape,  # (c, t, h, w, z, 2)
                                    'padding_left': 0,
                                    'padding_right': num_slices,
                                    'masks': patient_dir.replace(f"FullSample_test", f"Mask_Task2")
                                }

                                # Get the mask directory from attributes
                                mask_dir = attrs['masks']
                                modality_view = Path(full_ks_file_path).stem.replace('.h5', '')
                                
                                # Divide image into individual slices
                                for slice_ind in range(num_slices):
                                    
                                    # Filter the available masks for the current modality_view
                                    matching_masks = [mask_file for mask_file in os.listdir(mask_dir) if modality_view in mask_file]
                                    
                                    if matching_masks:
                                        # Randomly choose one mask from the matching masks
                                        chosen_mask = random.choice(matching_masks)
                                        mask_path = os.path.join(mask_dir, chosen_mask)

                                        # Add the sample with the selected mask
                                        self.raw_samples.append(
                                            CMRxReconRawDataSample(
                                                full_ks_file_path, 
                                                slice_ind, 
                                                attrs, 
                                                mask_path
                                            )
                                        )

                                # NOTE: This is the code to use all the available masks for each slice

                                # # Divide image into individual slices
                                # for slice_ind in range(num_slices):
                                    
                                #     # Each slice will have multiple masks, filter by modality_view
                                #     for mask_file in os.listdir(mask_dir):
                                #         if modality_view in mask_file:  # Filter by modality_view
                                #             mask_path = os.path.join(mask_dir, mask_file)
                                #             self.raw_samples.append(
                                #                 CMRxReconRawDataSample2024(
                                #                     full_ks_file_path, 
                                #                     slice_ind, 
                                #                     data_type, 
                                #                     mask_path
                                #                 )
                                #             )


            # Cache the dataset if not already cached
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
        fname, slice_ind, attrs, mask_path = self.raw_samples[i]

        kdata = utils.load_kdata(fname)

        # for ease we will rearrange (nt, nz, nc, ny, nx, 2) --> (nc, nt, ny, nx, nz, 2) 
        kdata = kdata.transpose(2,0,3,4,1,5)
        
        # select the kspace slice: (nc, nt, nx, ny, 2)
        kspace_data = np.asarray(kdata[:, :, :, :, slice_ind, :])

        # Extract modality_view and mask_info using split
        mask_fname = Path(mask_path).stem.replace('.mat', '')
        modality_view, mask_info = mask_fname.split('_mask_kt')

        # Determine the undersampling pattern, corresponding ACS region, and acceleration factor
        if 'Uniform' in mask_info:
            undersampling_pattern = 'Uniform'
            num_low_frequencies = [16] 
            mask_info = mask_info.replace('Uniform', '')
        elif 'Gaussian' in mask_info:
            undersampling_pattern = 'Gaussian'
            num_low_frequencies = [16] 
            mask_info = mask_info.replace('Gaussian', '')
        elif 'Radial' in mask_info:
            undersampling_pattern = 'Radial'
            num_low_frequencies = [16, 16] 
            mask_info = mask_info.replace('Radial', '')
        else:
            undersampling_pattern = None
            num_low_frequencies = []

        # The remaining part of mask_info is the acceleration factor
        acceleration_factor = int(mask_info)

        # add the mask specific data to the attributes
        attrs.update({
            'modality_view': modality_view,
            'acceleration_factor': acceleration_factor,
            'undersampling_pattern': undersampling_pattern,
            'num_low_frequencies': num_low_frequencies,
            'slice_ind': slice_ind,
            'fname': fname
        })

        with h5py.File(str(mask_path),'r') as hf:
            # just selects the first dataset in the hf file
            keys = list(hf.keys())[0]
            mask = np.asarray(hf[keys])

        if self.transform is None:
            sample = (kspace_data, mask, attrs, str(fname), self.max_img_size)
        else:
            sample = self.transform(kspace_data, mask, attrs, str(fname), self.max_img_size)

        return sample
    
