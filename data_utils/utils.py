import h5py
import math
import torch
import numpy as np
import fastmri
from fastmri.data import transforms as T

def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return shape [t,z,c,y,x,2], [t,z,y,x]
    '''
    kdata = load_kdata(filename)

    kdata_tensor = T.to_tensor(kdata).cuda()        # Convert from numpy array to pytorch tensor
    image_complex = fastmri.ifft2c(kdata_tensor)    # Apply Inverse Fourier Transform to get the complex image
    image_abs = fastmri.complex_abs(image_complex)  # Compute absolute value to get a real image
    image_rss = fastmri.rss(image_abs, dim=2)       # Compute the composite image using RSS
    
    image_np = image_rss.cpu().numpy()

    return kdata, image_np


def load_kdata(filename):
    '''
    Load k-space data from a .mat file.
    Return shape: [nt, nz, nc, ny, nx, 2]  # Note: 2 for real and imaginary parts
    '''
    # Load the .mat file
    data = loadmat(filename)
    
    # Access the k-space data using the known key
    kdata_real = data['kspace_full']['real']
    kdata_imag = data['kspace_full']['imag']
    
    # Stack real and imaginary parts along the last dimension
    kdata_combined = np.stack((kdata_real, kdata_imag), axis=-1)
    
    return kdata_combined


def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data


def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


def extract_number(filename):
    '''
    extract number from filename
    '''
    return ''.join(filter(str.isdigit, filename))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )

def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )

############# help function #############
def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2), 
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1], indices[3][0]:indices[3][1]]

def crop_submission(a, ismap=False):
    sx,sy,sz,st = a.shape
    if sz>=3:
        a = a[:,:,matlab_round(sz/2)-2:matlab_round(sz/2)]

    if ismap:
        b = _crop(a,(matlab_round(sx/3), matlab_round(sy/2),2,st))
    else:
        b = _crop(a[...,0:3],(matlab_round(sx/3), matlab_round(sy/2),2,3)) 
    return b
