#!/bin/bash -l
#SBATCH -J update_mask_paths
#SBATCH --time=06:00:00             # Adjust the time limit as necessary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # Adjust the number of CPUs if needed
#SBATCH --mem=32G                   # Adjust memory allocation as needed
#SBATCH --account=def-punithak      # Replace with your account name
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca  # Replace with your email
#SBATCH --output=slurm_logs/out/update_mask_paths_%j.out
#SBATCH --error=slurm_logs/err/update_mask_paths_%j.err

# Set environment variables
export HDF5_USE_FILE_LOCKING=FALSE

# Load necessary modules
module load python/3.10  # Adjust if you're using a different Python version

# Create and activate a virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install necessary Python packages
pip install --no-index h5py

# Run the Python script
srun python /home/nicocarp/scratch/CMRxRecon/scripts/update_masks.py