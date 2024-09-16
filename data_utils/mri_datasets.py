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

class CMRxReconRawDataSample2024(NamedTuple):
    fname: Path
    slice_ind: int
    dataset: str
    mask: Path

class CombinedCmrxReconSliceDataset2024(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        challenges: Sequence[str],
        data_partitions: Sequence[str],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "/home/nicocarp/scratch/CMR-Reconstruction/outputs/dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None   
    ):
        """
        Args:
            roots: Paths to the datasets.
            challenges: "singlecoil" or "multicoil" depending on which challenge to use.
            data_partitions: A sequence of "train", "val", or "test" depending on 
                which part of the dataset is being looked at for each dataset.
            transforms: Optional; A sequence of callable objects that preprocesses the raw data into appropriate form.
            sample_rates: Optional; A sequence of floats between 0 and 1.
            volume_sample_rates: Optional; A  sequence of floats between 0 and 1.
            use_dataset_cache: Whether to cache dataset metadata.
            dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
            num_cols: Optional; If provided, only slices with the desired number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample metadata as input and returns a boolean indicating whether the raw_sample should be included in the dataset.
        """

        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError("either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both")
        
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
            raise ValueError("Lengths of roots, transforms, challenges, sample_rates do not match")

        self.datasets = []
        self.raw_samples: List[CMRxReconRawDataSample2024] = []
        for i in range(len(roots)):
            self.datasets.append(
                CmrxReconSliceDataset2024(
                    root=roots[i],
                    transform=transforms[i],
                    challenge=challenges[i],
                    data_partition=data_partitions[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                    raw_sample_filter=raw_sample_filter
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
        data_partition: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "/home/nicocarp/scratch/CMR-Reconstruction/outputs/dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None 
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge to use.
            data_partition: "train", "val", or "test" depending on which part of the dataset is being looked at.
            transform: Optional; A callable object that pre-processes the raw data into appropriate form.
            use_dataset_cache: Whether to cache dataset metadata.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction of the slices should be loaded. Defaults to 1 if no value is given.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction of the volumes should be loaded. Defaults to 1 if no value is given.
            dataset_cache_file: Optional; A file in which to cache dataset information for faster load times.
            num_cols: Optional; If provided, only slices with the desired number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample metadata as input and returns a boolean indicating whether the raw_sample should be included in the dataset.
            data_partition: "train", "val", or "test" depending on which part of the dataset is being looked at.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both")

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform

        self.recons_key = "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # Set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # Load dataset cache if available and requested
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # Keep track of the max image size for regularization of inputs
        # used extra padding for convienient work in the cnns and transformer
        self.max_img_size = [32, 256, 512]

        # Check if dataset is in cache
        if dataset_cache.get(root) is None or not use_dataset_cache:
            
            data_type_list = ['Aorta', 'Cine', 'Mapping', 'Tagging']

            # Iterate through all the modalities
            for data_type in data_type_list:
                task_dir = os.path.join(root, data_type, 'TrainingSet', 'h5_FullSample', data_partition)

                # Iterate through all of the patients
                for patient_dir_info in os.scandir(task_dir):
                    if patient_dir_info.is_dir() and patient_dir_info.name not in ['.', '..', '.DS_Store']:
                        patient_dir = patient_dir_info.path

                        """# NOTE: for testing dataset creation, remove after
                        last_part = patient_dir.name  # This gets the last directory name (e.g., 'P002')

                        # Get the last character of the last directory name and convert it to an int
                        if last_part[-1].isdigit():
                            last_char_as_int = int(last_part[-1])

                            # Perform your check on the last integer
                            if last_char_as_int > 2: 
                                continue"""
                            
                        # Iterate through all of the views/weightings/etc. for this patient
                        for ks_file_info in os.scandir(patient_dir):
                            if ks_file_info.is_file() and ks_file_info.name not in ['.', '..', '.DS_Store']:
                                full_ks_file_path = ks_file_info.path  

                                # Open and process the file
                                with h5py.File(full_ks_file_path, 'r') as hf:
                                    x, y, c, t, num_slices = hf["kspace"].shape
                                    # Shape of k-space: [nx, ny, nc, nz, nt]

                                    # Get the mask directory from attributes
                                    mask_dir = hf.attrs['masks']
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
                                                CMRxReconRawDataSample2024(
                                                    full_ks_file_path, 
                                                    slice_ind, 
                                                    data_type, 
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

        # Subsample if desired
        if sample_rate < 1.0:  # Sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # Sample by volume
            vol_names = sorted(list(set([f.fname.stem for f in self.raw_samples])))
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
                if ex.attrs["encoding_size"][1] in num_cols  # type: ignore
            ]
    
    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i: int):
        fname, slice_ind, data_type, mask_path = self.raw_samples[i]

        with h5py.File(str(fname), 'r') as hf:
            kspace_data = np.asarray(hf["kspace"][:, :, :, :, slice_ind])
            kspace_data = kspace_data.transpose(2, 3, 0, 1) # (x, y, c, t) --> (c, t, x, y)

            # NOTE: come back to this, issue with preprocssing data caused issues with t dim
            """
            if self.recons_key in hf:
                target_data = np.asarray(hf[self.recons_key][:, :, slice_ind, :])
                target_data = target_data.transpose(2, 0, 1)
            else:
                target_data = None
            """
            
            # apply the inverse fourier transform to the kspace data to send put it in the image domain
            kdata_tensor = torch.tensor(kspace_data)
            target_data_coils = torch.fft.ifft2(kdata_tensor, dim=(-2, -1), norm='ortho')    
            
            # Sum over coil dimension to get the composite coil image
            target_data = torch.linalg.vector_norm(target_data_coils, ord=2, dim=0, keepdim=True)  
            target_data = target_data.numpy()

            attrs = dict(hf.attrs)

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


        attrs.update({
            'modality_view': modality_view,
            'acceleration_factor': acceleration_factor,
            'undersampling_pattern': undersampling_pattern,
            'num_low_frequencies': num_low_frequencies,
            'slice_ind': slice_ind,
            'fname': fname
        })

        with h5py.File(str(mask_path),'r') as hf:
            keys = list(hf.keys())[0]
            mask = np.asarray(hf[keys])

        if self.transform is None:
            sample = (kspace_data, mask, target_data, attrs, str(fname), self.max_img_size)
        else:
            sample = self.transform(kspace_data, mask, target_data, attrs, str(fname), self.max_img_size)

        return sample


#########################################################################################################
# CMRxRecon 2023 part
#########################################################################################################

class CMRxReconRawDataSample2023(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]

class CombinedCmrxReconSliceDataset2023(torch.utils.data.Dataset):
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
        self.raw_samples: List[CMRxReconRawDataSample2023] = []
        for i in range(len(roots)):
            self.datasets.append(
                CmrxReconSliceDataset2023(
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

class CmrxReconSliceDataset2023(torch.utils.data.Dataset):
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
        num_adj_slices = 5 #15
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

        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset" 
        self.num_adj_slices = num_adj_slices

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
            folders = list(Path(root).iterdir())
            files = []
            for i in range(len(folders)):
                files += list(folders[i].iterdir())
            # include mapping data 
            folders = list(Path(str(root).replace('Cine','Mapping')).iterdir())
            files2 = []
            for i in range(len(folders)):
                files2 += list(folders[i].iterdir())
            files = sorted(files+files2)

            for fname in sorted(files): 
                with h5py.File(fname,'r') as hf:
                    num_slices = hf["kspace"].shape[0]
                    metadata = { **hf.attrs }
                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = CMRxReconRawDataSample2023(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)
                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # balance the different type of training data 
        if 'train' in str(root):
            raw_samples_t1 = []
            raw_samples_t2 = []
            raw_samples_lax = []
            raw_samples_sax = []
            for ii in self.raw_samples:
                fname_ii = str(ii.fname)
                if 'T1' in fname_ii:
                    raw_samples_t1.append(ii)
                elif 'T2' in fname_ii:
                    raw_samples_t2.append(ii)
                elif 'lax' in fname_ii:
                    raw_samples_lax.append(ii)
                elif 'sax' in fname_ii:
                    raw_samples_sax.append(ii)
            self.raw_samples = raw_samples_t1*2 + raw_samples_t2*5 + raw_samples_lax*3 + raw_samples_sax
        # self.raw_samples = self.raw_samples[0:1000]  # for quick debug

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
    
    def _get_frames_indices(self, dataslice, num_slices_in_volume, num_t_in_volume=None, is_lax=False):
        '''
        when we reshape t, z to one axis in preprocessing, we need to get the indices of the slices in the original t, z axis;
        then find the adjacent slices in the original z axis
        '''
        ti = dataslice//num_slices_in_volume
        zi = dataslice - ti*num_slices_in_volume

        zi_idx_list = [zi]

        ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-2,3)]
        output_list = []

        for zz in zi_idx_list:
            for tt in ti_idx_list:
                output_list.append(tt*num_slices_in_volume + zz)

        return output_list
    def _get_frames_indices_mapping(self, dataslice, num_slices_in_volume, num_t_in_volume=None, isT2=False):
        '''
        when we reshape t, z to one axis in preprocessing, we need to get the indices of the slices in the original t, z axis;
        then find the adjacent slices in the original z axis
        '''
        ti = dataslice//num_slices_in_volume
        zi = dataslice - ti*num_slices_in_volume

        zi_idx_list = [zi]

        if isT2: # only 3 nw in T2, so we repeat adjacent for 3 times
            ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-1,2)]
            ti_idx_list = 1*ti_idx_list[0:1] + ti_idx_list + ti_idx_list[2:3]*1
        else:
            ti_idx_list = [ (i+ti)%num_t_in_volume for i in range(-2,3)]
        output_list = []

        for zz in zi_idx_list:
            for tt in ti_idx_list:
                output_list.append(tt*num_slices_in_volume + zz)

        return output_list

    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        
        #TODO: use metadata to decide rather than fname is better
        islax=False if 'sax' in fname.name else True
        isT2=True if 'T2' in fname.name else False

        kspace = []
        with h5py.File(str(fname),'r') as hf:
            kspace_volume = hf["kspace"]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)

            num_slices = attrs['shape'][1]
            num_t = attrs['shape'][0]
            if 'Cine' in str(fname):
                slice_idx_list = self._get_frames_indices(dataslice, num_slices,num_t, is_lax=islax) 
            else:
                slice_idx_list = self._get_frames_indices_mapping(dataslice, num_slices,num_t,isT2=isT2)  
            for idx in slice_idx_list:
                kspace.append(kspace_volume[idx])
            kspace = np.concatenate(kspace, axis=0)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, str(fname), dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, str(fname), dataslice)

        return sample


#########################################################################################################
# fastmri part
#########################################################################################################

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

class FastmriKneeRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]

class CombinedFastmriKneeSliceDataset(torch.utils.data.Dataset):
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
        self.raw_samples: List[FastmriKneeRawDataSample] = []
        for i in range(len(roots)):
            self.datasets.append(
                FastmriKneeSliceDataset(
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

class FastmriKneeSliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

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

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastmriKneeRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

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

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs,
            }

        return metadata, num_slices

    def __len__(self):
        return len(self.raw_samples)
    
    def _get_frames_indices(self,dataslice, num_slices):
        z_list = [ min(max(i+dataslice,0), num_slices-1) for i in range(-1,2)]
        return z_list

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]
        
        kspace = []
        with h5py.File(fname, "r") as hf:
            num_slices = hf["kspace"].shape[0]
            slice_idx_list = self._get_frames_indices(dataslice, num_slices)
            for slice_idx in slice_idx_list:
                kspace.append(hf["kspace"][slice_idx])
            kspace = np.concatenate(kspace, axis=0)


            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return sample