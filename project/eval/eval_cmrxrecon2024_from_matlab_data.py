import os
import sys
import pathlib
import argparse
import yaml
import torch
from torch import nn
import scipy.io
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matlab.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))
import fastmri.data.transforms as T
from models.jCAN.model import ReconModel
from models.jCAN.utils import pad_to_max_size, unpad_from_max_size, create_padding_mask
from data_utils.cmrxrecon2024.utils import load_kdata, loadmat
from data_utils.cmrxrecon2024.utils import count_parameters, count_trainable_parameters, count_untrainable_parameters

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fname, num_low_frequencies):
        self.fname = fname
        self.kspace = load_kdata(fname)
        self.num_low_frequencies = num_low_frequencies
        self.num_files = self.kspace.shape[0]

    def __getitem__(self, dataslice):
        kspace_torch = T.to_tensor(self.kspace[dataslice])
        masked_kspace = kspace_torch
        return masked_kspace, dataslice

    def __len__(self):
        return self.num_files

def reconstruct_volume(volume_slices, slice_indices):
    sorted_slices = [volume_slices[i] for i in sorted(slice_indices)]

    #TODO: check thta this is being stacked across the correct axis
    return np.stack(sorted_slices, axis=-3)

def get_undersampling_info_from_filename(filename):
    # Determine the undersampling pattern and corresponding ACS region
    if 'Uniform' in filename:
        undersampling_pattern = 'Uniform'
        num_low_frequencies = [16] 
    elif 'Gaussian' in filename:
        undersampling_pattern = 'Gaussian'
        num_low_frequencies = [16] 
    elif 'Radial' in filename:
        undersampling_pattern = 'Radial'
        num_low_frequencies = [16, 16] 
    else:
        undersampling_pattern = None
        num_low_frequencies = None

    return undersampling_pattern, num_low_frequencies

def predict(config):
    device = 'cuda:0'
    model_config = config['model']
    data_config = config['data']
    center_crop = data_config['center_crop']

    # Load model
    model = ReconModel(
        coils=model_config['coils'],
        num_heads=model_config['num_heads'],
        window_size=model_config['window_size'],
        depths=model_config['depths'],
        patch_size=model_config['patch_size'],
        embed_dim=model_config['embed_dim'],
        mlp_ratio=model_config['mlp_ratio'],
        qkv_bias=model_config['qkv_bias'],
        qk_scale=model_config['qk_scale'],
        drop=model_config['drop'],
        attn_drop=model_config['attn_drop'],
        drop_path=model_config['drop_path'],
        norm_layer=model_config['norm_layer'],
        n_SC=model_config['n_SC'],
        num_recurrent=model_config['num_recurrent'],
        sens_chans=model_config['sens_chans'],
        sens_steps=model_config['sens_steps'],
        scale=model_config['scale']
    )

    print(f'Model:\ntotal param: {count_parameters(model)}\ntrainable param: {count_trainable_parameters(model)}\nuntrainable param: {count_untrainable_parameters(model)}\n##############')

    state_dict = torch.load(data_config['model_path'])['state_dict']
    state_dict.pop('loss.w')
    state_dict = {k.replace('promptmr.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Prepare the file list
    input_dir = data_config['input_dir']
    evaluate_set = data_config['evaluate_set']

    under_aorta_matlab_folder = os.path.join(input_dir, f"Aorta/{evaluate_set}/Undersample_Task2")
    under_cine_matlab_folder = os.path.join(input_dir, f"Cine/{evaluate_set}/Undersample_Task2")
    under_mapping_matlab_folder = os.path.join(input_dir, f"Mapping/{evaluate_set}/Undersample_Task2")
    under_tagging_matlab_folder = os.path.join(input_dir, f"Tagging/{evaluate_set}/Undersample_Task2")

    f_aorta = sorted(glob.glob(os.path.join(under_aorta_matlab_folder, '**/*.mat'), recursive=True))
    f_cine = sorted(glob.glob(os.path.join(under_cine_matlab_folder, '**/*.mat'), recursive=True))
    f_mapping = sorted(glob.glob(os.path.join(under_mapping_matlab_folder, '**/*.mat'), recursive=True))
    f_tagging = sorted(glob.glob(os.path.join(under_tagging_matlab_folder, '**/*.mat'), recursive=True))

    f = f_cine + f_mapping + f_aorta + f_tagging
    print(f'##############\n Total files: {len(f)}\n##############')

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Predict
    with torch.no_grad():
        for ff in tqdm(f, desc='files'):
            print('-- processing --', ff)

            # Get undersampling information from the filename
            undersampling_pattern, num_low_frequencies = get_undersampling_info_from_filename(ff)

            dataset = Dataset(ff, num_low_frequencies)
            dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'], pin_memory=True, drop_last=False)
            volume_slices = {}
            slice_indices = []

            for masked_kspace, dataslice in tqdm(dataloader, desc='stage1'):
                bs = masked_kspace.shape[0]

                # Pad inputs to the maximum image size
                max_img_size = tuple(data_config['max_img_size'])
                masked_kspace_padded, pad_params_kspace = pad_to_max_size(masked_kspace, max_img_size)
                padding_mask = create_padding_mask(masked_kspace, max_img_size)

                output = model(masked_kspace_padded.to(device), padding_mask=padding_mask.to(device))

                # Unpad the output
                output_unpadded = unpad_from_max_size(output.cpu(), masked_kspace.shape[-4:])

                for i in range(bs):
                    slice_idx = dataslice[i].item()
                    volume_slices[slice_idx] = output_unpadded[i:i+1].numpy()
                    slice_indices.append(slice_idx)

            volume_result = reconstruct_volume(volume_slices, slice_indices)

            # Extracting modality and patient ID from file path
            modality = os.path.basename(os.path.dirname(os.path.dirname(ff)))
            patient_id = os.path.basename(os.path.dirname(ff))
            file_name = os.path.basename(ff)

            save_path = os.path.join(data_config['output_dir'], f"{modality}/ValidationSet/Task2/{patient_id}/{file_name}")
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if center_crop:
                # Call MATLAB function run4Ranking
                img = matlab.double(volume_result.tolist())
                img4ranking = eng.run4Ranking(img, file_name)

                scipy.io.savemat(save_path, {'img4ranking': np.array(img4ranking)})
            else:
                scipy.io.savemat(save_path, {'img4ranking': volume_result})

            print('-- saving --', save_path)

    eng.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Input data store in:", config['data']['input_dir'])
    print("Output data store in:", config['data']['output_dir'])

    predict(config)
