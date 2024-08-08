import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import shutil
import argparse
import numpy as np
from utils import zf_recon
import h5py
import glob
from os.path import join
from tqdm import tqdm

def split_train_val(h5_folder, train_num=160):
    """
    Split the data into training and validation sets.
    Args:
        h5_folder (str): The folder containing the h5 files.
        train_num (int): The number of cases to be used for training.
    """
    train_folder = join(h5_folder, 'train')
    val_folder = join(h5_folder, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    if os.path.exists(h5_folder):
        num_folders = len(os.listdir(h5_folder))

        for i in range(1, num_folders+1):
            case_folder = join(h5_folder, f"P{i:03d}")
            
            if os.path.exists(case_folder):
                if i <= train_num:
                    shutil.move(case_folder, train_folder)
                else:
                    shutil.move(case_folder, val_folder)

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
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        filename = os.path.basename(ff)

        # <-- ff:    (x, y, c, z, t)
        # --> kdata: (t, z, c, y, x)
        # --> image: (t, z, y, x)
        kdata, image = zf_recon(ff)     

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset 
        # kdata is of shape time, slice, coil, phase_enc, readout (t, z, c, y, x)
        # transpose to slice, time, coil, readout, phase_enc (z, t, c, x, y) as we plan to batch by the slices in fastMRI style
        save_kdata = kdata.reshape.transpose(1,0,2,4,3)
        file.create_dataset('kspace', data=save_kdata)

        # image is of shape time, slice, coil, phase_enc, readout (t, z, y, x)
        # transpose to slice, time, coil, readout, phase_enc (z, t, x, y) 
        save_image = image.transpose(1,0,3,2)
        file.create_dataset('reconstruction_rss', data=save_image)
        file.attrs['max'] = image.max()
        file.attrs['norm'] = np.linalg.norm(image)

        # Add attributes to the dataset
        if 'T1' in filename:
            file.attrs['acquisition'] = 'FLASH-T1'
        elif 'T2' in filename:
            file.attrs['acquisition'] = 'FLASH-T2'
        elif 'lax' in filename:
            file.attrs['acquisition'] = 'TrueFISP-LAX'
        elif 'sax' in filename:
            file.attrs['acquisition'] = 'TrueFISP-SAX'
        elif 'lvot' in filename:
            file.attrs['acquisition'] = 'TrueFISP-LVOT'
        elif 'tagging' in filename:
            file.attrs['acquisition'] = 'TrueFISP-Tagging'
        elif 'tra' in filename:
            file.attrs['acquisition'] = 'TrueFISP-Aorta_TRA'
        elif 'sag' in filename: 
            file.attrs['acquisition'] = 'TrueFISP-Aorta_SAG'
        else:
            raise ValueError('unknown acquisition type')

        file.attrs['patient_id'] = save_path.split(save_folder_name)[-1][0:3]   # This will isolate the patient id P### 
        file.attrs['shape'] = save_kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[4]
        file.attrs['encoding_size'] = (save_kdata.shape[3], save_kdata.shape[4], 1)
        file.attrs['recon_size'] = (save_kdata.shape[3], save_kdata.shape[4], 1)

        file.attrs['masks'] = join(save_path.replace(save_folder_name, f"Mask_Task{task}").split(save_folder_name)[0], file.attrs['patient_id'])

        # Close the file
        file.close()
    
    # 2. Split first 160 cases as training set and the rest 40 cases as validation set
    aorta_h5_folder = fully_aorta_matlab_folder.replace('FullSample', save_folder_name)
    cine_h5_folder = fully_cine_matlab_folder.replace('FullSample', save_folder_name)
    mapping_h5_folder = fully_mapping_matlab_folder.replace('FullSample', save_folder_name)
    tagging_h5_folder = fully_tagging_matlab_folder.replace('FullSample', save_folder_name)

    split_train_val(aorta_h5_folder, train_num=160)
    split_train_val(cine_h5_folder, train_num=160)
    split_train_val(mapping_h5_folder, train_num=160)
    split_train_val(tagging_h5_folder, train_num=160)
