#!/bin/bash -l
#SBATCH -J cmrxrecon_train
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # 4 tasks per node (1 per GPU)
#SBATCH --gpus-per-node=v100l:1            # 4 GPUs per node
#SBATCH --cpus-per-task=8                  # Request 8 CPUs per task (adjust if needed)
#SBATCH --mem=128GB                        # Request 128GB of memory                            
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

# Set PROJECT_ROOT and PYTHONPATH
export PROJECT_ROOT=/home/nicocarp/scratch/CMR-Reconstruction
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


config=$PROJECT_ROOT/configs/train_config.yaml

module purge

# Load necessary modules
module load matlab/2023b
module load StdEnv/2023
module load gcc/12.3
module load hdf5/1.14.2
module load cuda/12.2
module load cudnn/8.9.5.29
module load nccl/2.18.3
module load python/3.10

# Create and activate a virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages from Compute Canada wheels
pip install --no-index -r $PROJECT_ROOT/configs/requirements_local.txt

# Install packages from PyPI or other sources
pip install -r $PROJECT_ROOT/configs/requirements_pypi.txt

# Verify the configuration file exists
if [ ! -f "$config" ]; then
    echo "Configuration file $config not found!"
    exit 1
fi

srun python $PROJECT_ROOT/data_utils/cmrxrecon2024/preprocess_cmrxrecon2024.py --data_path $PROJECT_ROOT/datasets/CMR_2024/ChallengeData/MultiCoil/ --h5py_folder h5_FullSample/


