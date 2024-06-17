<div align="center">    
 
# CMR Reconstruction

</div>


## Introduction

**Contact**
Nicolas Carpenter: ngcarpen@ualberta.ca


## 1. Description
This repository implements the  
- Our code offers the following things:
  


## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/NicoCarpe/CMR-Reconstruction.git

# install project   
cd CMR-Reconstruction
conda env create -f envs/py39.yaml
conda activate py39
 ```

## 3. Project Structure
Our directory structure looks like this:

```
├── output                       <- Experiment log and checkpoints will be saved here once you train a model
├── envs                         <- Conda environment
├── pretrained_models            <- Pretrained model checkpoints
├── project                 
│   ├── module                   <- Every module is given in this directory
│   │   ├── models               <- Models (Swin fMRI Transformer)
│   │   ├── utils                
│   │   │    ├── data_module.py  <- Dataloader & codes for matching CMR scans and target variables
│   │   │    └── data_preprocessing_and_load
│   │   │        ├── datasets.py           <- Dataset Class for each dataset
│   │   │        └── preprocessing.py      <- Preprocessing codes for step 6
│   │   └── pl_classifier.py    <- LightningModule
│   └── main.py                 <- Main code that trains and tests the model
│ 
├── scripts                     <- shell scripts for training
│
├── .gitignore                  <- List of files/folders ignored by git
├── export_DDP_vars.sh          <- setup file for running torch DistributedDataParallel (DDP) 
└── README.md
```

<br>

## 4. Train model

### 4.0 Quick start & Tutorial

### 4.1 Arguments for trainer

### 4.2 Hidden Arguments for PyTorch lightning

### 4.3 Commands/scripts for running tasks

## 5. Loggers

## 6. How to prepare your own dataset
These preprocessing codes are implemented based on the initial repository by GonyRosenman [TFF](https://github.com/GonyRosenman/TFF)

To make your own dataset, you should execute either of the minimal preprocessing steps:

After the minimal preprocessing steps, you should perform additional preprocessing to use SwiFT. (You can find the preprocessing code at 'project/module/utils/data_preprocessing_and_load/preprocessing.py')
- normalization: voxel normalization(not used) and whole-brain z-normalization (mainly used)
- change fMRI volumes to floating point 16 to save storage and decrease IO bottleneck.
- each fMRI volume is saved separately as torch checkpoints to facilitate window-based training.
- remove non-brain(background) voxels that are over 96 voxels.
   - you should open your fMRI scans to determine the level that does not cut out the brain regions
   - you can use `nilearn` to visualize your fMRI data. (official documentation: [here](https://nilearn.github.io/dev/index.html))
  ```python
  from nilearn import plotting
  from nilearn.image import mean_img
  
  plotting.view_img(mean_img(fmri_filename), threshold=None)
  ```
   - if your dimension is under 96, you can pad non-brain voxels at 'datasets.py' files.

* refer to the annotation in the 'preprocessing.py' code to adjust it for your own datasets.

The resulting data structure is as follows:
```
├── {Dataset name}_MNI_to_TRs                 
   ├── img                  <- Every normalized volume is located in this directory
   │   ├── sub-01           <- subject name
   │   │  ├── frame_0.pt    <- Each torch pt file contains one volume in a fMRI sequence (total number of pt files = length of fMRI sequence)
   │   │  ├── frame_1.pt
   │   │  │       :
   │   │  └── frame_{T}.pt  <- the last volume in an fMRI sequence (length T) 
   │   └── sub-02              
   │   │  ├── frame_0.pt    
   │   │  ├── frame_1.pt
   │   │  ├──     :
   └── metadata
       └── metafile.csv     <- file containing target variable
```

## 7. Define the Dataset class for your own dataset.
* The data loading pipeline works by processing image and metadata at 'project/module/utils/data_module.py' and passing the paired image-label tuples to the Dataset classes at 'project/module/utils/data_preprocessing_and_load/datasets.py.'
* you should implement codes for combining image path, subject_name, and target variables at 'project/module/utils/data_module.py'
* you should define the Dataset Class for your dataset at 'project/module/utils/data_preprocessing_and_load/datasets.py.' In the Dataset class (__getitem__), you should specify how many background voxels you would add or remove to make the volumes shaped 96 * 96 * 96.

## 8. Pretrained model checkpoints



### Citation   
```

```   
