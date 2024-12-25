import os
import h5py
import glob
from os.path import join, basename, dirname

def update_mask_paths(base_folder, task):
    # Define the folders containing only .h5 files
    # fully_aorta_h5_folder = join(base_folder, "Aorta/TrainingSet/h5_FullSample")
    # fully_cine_h5_folder = join(base_folder, "Cine/TrainingSet/h5_FullSample")
    # fully_mapping_h5_folder = join(base_folder, "Mapping/TrainingSet/h5_FullSample")
    fully_tagging_h5_folder = join(base_folder, "Tagging/TrainingSet/h5_FullSample")

    # Initialize list to store all .h5 file paths
    h5_files = []

    # Iterate over each modality folder
    for folder in [fully_tagging_h5_folder]:    # fully_aorta_h5_folder, fully_cine_h5_folder, fully_mapping_h5_folder, 
        # Iterate over 'train' and 'val' subdirectories
        for subfolder in ['train', 'val']:
            # Create the full path to the 'train' or 'val' directory
            subfolder_path = join(folder, subfolder)
            
            # Use glob to find all .h5 files recursively in these subdirectories
            found_files = glob.glob(join(subfolder_path, '**/*.h5'), recursive=True)
            h5_files.extend(found_files)
            print(f"Found {len(found_files)} files in {subfolder_path}", flush=True)

    # Print total number of files found
    print(f"Total .h5 files found: {len(h5_files)}", flush=True)

    # Iterate over the combined list of .h5 files
    for h5_file in h5_files:

        # Open the .h5 file
        with h5py.File(h5_file, 'r+') as f:
            # Get the patient ID (e.g., P002)
            patient_id = basename(dirname(h5_file))

            # Split the path at 'h5_FullSample' to get the base path
            base_path = h5_file.split('h5_FullSample')[0]

            # Construct the correct masks path using f-string
            masks_path = join(base_path, f"Mask_Task{task}", patient_id)

            # Update the masks attribute in the HDF5 file
            f.attrs['masks'] = masks_path

            print(f"Updated masks path: {masks_path} for {h5_file}", flush=True)

if __name__ == '__main__':
    base_folder = "/home/nicocarp/scratch/CMRxRecon/datasets/CMR_2024/ChallengeData/MultiCoil/"
    task = 2 

    update_mask_paths(base_folder, task)
