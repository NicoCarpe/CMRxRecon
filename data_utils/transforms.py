from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from data_utils.subsample import MaskFunc
import fastmri
from fastmri import fft2c, ifft2c


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: Tuple[int, int], mask_to: Tuple[int, int]) -> torch.Tensor:
    """
    Initializes a mask with the center filled in for both `h` and `w` dimensions.

    Args:
        mask_from: Tuple with starting points for `h` and `w` dimensions.
        mask_to: Tuple with ending points for `h` and `w` dimensions.

    Returns:
        A mask with the center filled in both `h` and `w` dimensions.
    """
    mask = torch.zeros_like(x)
    h_from, w_from = mask_from
    h_to, w_to = mask_to
    mask[:, :, :, h_from:h_to, w_from:w_to] = x[:, :, :, h_from:h_to, w_from:w_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in for each batch element in both `h` and `w` dimensions.

    Args:
        x: Input tensor to be masked.
        mask_from: Tensor with starting points for `h` and `w` dimensions for each batch.
        mask_to: Tensor with ending points for `h` and `w` dimensions for each batch.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not (mask_from.ndim == 2 and mask_from.shape[1] == 2):
        raise ValueError("mask_from and mask_to must have shape (batch_size, 2).")
    if not x.shape[0] == mask_from.shape[0]:
        raise ValueError("mask_from and mask_to must have batch_size length as the first dimension.")

    mask = torch.zeros_like(x)
    for i, ((h_from, w_from), (h_to, w_to)) in enumerate(zip(mask_from, mask_to)):
        mask[i, :, :, h_from:h_to, w_from:w_to] = x[i, :, :, h_from:h_to, w_from:w_to]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class MRISample(NamedTuple):
    """
    A sample of masked k-space for reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        target: The target image (if applicable).
        attrs: Image attributes
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    target_k: torch.Tensor
    attrs: Dict
    max_img_size: list

class MRIDataTransform:
    """
    Data Transformer for training model.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        attrs: Dict,
        fname: str,
        max_img_size: list
    ) -> MRISample:

        """
        Args:
            kspace: Input k-space with shape (c, t, h, w, 2)
            mask: Undersampling mask with shape (t, h, w)
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            max_img_size: Max image dimensions (t, h, w)

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the max image dimensions.
        """
        
        kspace_torch = torch.from_numpy(kspace)  # Direct conversion from numpy array to PyTorch tensor

        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = MRISample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),     # convert to boolean tensor for memory efficiency
                target_k=kspace_torch,
                attrs=attrs,
                max_img_size=max_img_size
            )
        else:
            # Get the shape of the k-space data 
            c, t, h, w, comp = kspace_torch.shape

            # Determine mask shape and adjust for interleaving
            if mask.ndim == 3:  # Temporal interleaving
                mask_shape = [1, t, h, w, 1]

            elif mask.ndim == 2:  # Spatial mask only
                mask_shape = [1, 1, h, w, 1]
            
            # Convert the numpy mask to a torch tensor and reshape it to the correct dimensions
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

            # Zero out regions outside the acquisition window in the mask
            # mask_torch[:, :, :, :acq_start, :] = 0
            # mask_torch[:, :, :, acq_end:, :] = 0

            # Apply the mask to k-space data
            masked_kspace = kspace_torch * mask_torch 

            sample = MRISample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),     # convert to boolean tensor for memory efficiency
                target_k=kspace_torch,
                attrs=attrs,
                max_img_size=max_img_size
            )

        return sample    