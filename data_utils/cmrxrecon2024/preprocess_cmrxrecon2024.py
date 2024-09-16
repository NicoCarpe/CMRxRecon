import os
import sys
import shutil
import argparse
import numpy as np
from data_utils.cmrxrecon2024.utils import zf_recon
import h5py
import glob
from os.path import join
from tqdm import tqdm

def split_train_val(h5_folder, train_num=100):
    train_folder = join(h5_folder, 'train')
    val_folder = join(h5_folder, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    case_count = 0

    # Iterate over all folders within the h5_folder, excluding 'train' and 'val'
    for case_folder in sorted(os.listdir(h5_folder)):
        full_case_folder = join(h5_folder, case_folder)

        # Skip if the folder is 'train' or 'val'
        if case_folder in ['train', 'val']:
            continue

        case_count += 1

        # Move the case folder to the appropriate train or val folder
        if case_count <= train_num:
            shutil.move(full_case_folder, train_folder)
        else:
            shutil.move(full_case_folder, val_folder)

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=int,
        default=2,
        help="The task in which our data is attributed to.",
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="~/scratch/CMR-Reconstruction/datasets/CMR_2024/ChallengeData/MultiCoil/",
        help="Path to the multi-coil MATLAB folder.",
    )

    parser.add_argument(
        "--h5py_folder",
        type=str,
        default="h5_FullSample/",
        help="The folder name to save the h5py files.",
    )

    args = parser.parse_args() 
    task = args.task
    data_path = args.data_path
    save_folder_name = args.h5py_folder

    fully_aorta_matlab_folder = join(data_path, "Aorta/TrainingSet/FullSample")
    fully_cine_matlab_folder = join(data_path, "Cine/TrainingSet/FullSample")
    fully_mapping_matlab_folder = join(data_path, "Mapping/TrainingSet/FullSample")
    fully_tagging_matlab_folder = join(data_path, "Tagging/TrainingSet/FullSample")

    assert os.path.exists(fully_aorta_matlab_folder), f"Path {fully_aorta_matlab_folder} does not exist."
    assert os.path.exists(fully_cine_matlab_folder), f"Path {fully_cine_matlab_folder} does not exist."
    assert os.path.exists(fully_mapping_matlab_folder), f"Path {fully_mapping_matlab_folder} does not exist."
    assert os.path.exists(fully_tagging_matlab_folder), f"Path {fully_tagging_matlab_folder} does not exist."

    # 0. Get input file list
    f_aorta = sorted(glob.glob(join(fully_aorta_matlab_folder, '**/*.mat'), recursive=True))
    f_cine = sorted(glob.glob(join(fully_cine_matlab_folder, '**/*.mat'), recursive=True))
    f_mapping = sorted(glob.glob(join(fully_mapping_matlab_folder, '**/*.mat'), recursive=True))
    f_tagging = sorted(glob.glob(join(fully_tagging_matlab_folder, '**/*.mat'), recursive=True))

    f = f_cine + f_mapping + f_aorta + f_tagging
    print('Total number of files: ', len(f))
    print('Aorta cases: ', len(os.listdir(fully_aorta_matlab_folder)), ', Aorta files: ', len(f_aorta))
    print('Cine cases: ', len(os.listdir(fully_cine_matlab_folder)), ', Cine files: ', len(f_cine))
    print('Mapping cases: ', len(os.listdir(fully_mapping_matlab_folder)), ', Mapping files: ', len(f_mapping))
    print('Tagging cases: ', len(os.listdir(fully_tagging_matlab_folder)), ', Tagging files: ', len(f_tagging))

    # 1. Save as fastMRI style h5py files
    for ff in tqdm(f):
        save_path = ff.replace('FullSample', save_folder_name).replace('.mat', '.h5')
        
        # Check if the processed file already exists
        if os.path.exists(save_path):
            #print(f"skip {save_path}")
            continue  # Skip to the next file

        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        filename = os.path.basename(ff)

        # Load k-space and image data using zf_recon (assuming output shape: kdata=(nx,ny,nc,nz,nt), image=(nx,ny,nz,nt))
        kdata, image = zf_recon(ff)

        # Save the data to an h5py file
        with h5py.File(save_path, 'w') as file:
            # TODO: these transposes were done poorly to fit the rest of the project
            
            # Transpose the k-space data to match fastMRI format
            # Original shape: (nx, ny, nc, nz, nt)
            # Transposed shape: (nc, nx, ny, nt, nz)
            save_kdata = kdata.transpose(2, 0, 1, 4, 3)
            
            # Transpose the image data to match fastMRI format
            # Original shape: (nx, ny, nz, nt)
            # Transposed shape: (nz, nx, ny, nt)
            save_image = image.transpose(2, 0, 1, 3)

            # Create datasets in the h5py file
            file.create_dataset('kspace', data=save_kdata)
            file.create_dataset('reconstruction_rss', data=save_image)

            # Compute and store attributes
            attrs = {
                'max': image.max(),
                'norm': np.linalg.norm(image),
                'patient_id': save_path.split(save_folder_name)[-1][0:3],
                'shape': save_kdata.shape,  # (nc, nx, ny, nt, nz)
                'padding_left': 0,
                'padding_right': save_kdata.shape[4],  # This refers to the nz dimension
                'encoding_size': (save_kdata.shape[3], save_kdata.shape[4], 1),  # (nt, nz, 1)
                'recon_size': (save_kdata.shape[3], save_kdata.shape[4], 1),  # (nt, nz, 1)
                'masks': save_path.replace(f"save_folder_name", f"Mask_Task{task}")
            }

            # Update file attributes
            file.attrs.update(attrs)

    # 2. Dynamically split cases into training and validation sets
    aorta_h5_folder = fully_aorta_matlab_folder.replace('FullSample', save_folder_name)
    cine_h5_folder = fully_cine_matlab_folder.replace('FullSample', save_folder_name)
    mapping_h5_folder = fully_mapping_matlab_folder.replace('FullSample', save_folder_name)
    tagging_h5_folder = fully_tagging_matlab_folder.replace('FullSample', save_folder_name)

    # Case Totals:

    # Aorta cases:  155
    # Cine cases:  196
    # Mapping cases:  193
    # Tagging cases:  143

    # Create a 160:50 train val split
    split_train_val(aorta_h5_folder, train_num=124)         
    split_train_val(cine_h5_folder, train_num=157)
    split_train_val(mapping_h5_folder, train_num=154)
    split_train_val(tagging_h5_folder, train_num=114)
