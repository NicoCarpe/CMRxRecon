#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH -J cmrxrecon_eval
#SBATCH --open-mode=append
#SBATCH -C h100

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

master_node=$SLURMD_NODENAME
ENV_DIR=$SCRATCH/cmrxrecon_env
config="eval_config.yaml"

module purge
module load gcc openmpi/4.0.7 python-mpi/3.11.2
module load hdf5/mpi-1.12.2
module load cuda/12.1 cudnn/cuda12-8.9.0 nccl/cuda12.1-2.18.1

# Create conda environment if it does not exist
if [ ! -d "$ENV_DIR" ]; then
    module load miniconda
    conda create -p $ENV_DIR --file env.yaml -y
fi

source activate $ENV_DIR

srun python eval.py --config $config
