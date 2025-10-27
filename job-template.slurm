#!/bin/bash
#SBATCH --job-name=testing-loop             # Job name

#SBATCH --account=samarjit                   # Replace with your netid
#SBATCH --partition=gpu                   # Partition name (default is gpu)
#SBATCH --nodes=1                         # One node (tomago/tempura)
#SBATCH --ntasks-per-node=1               # Number of CPU cores (vary it as needed)
#SBATCH --mem=16G                         # Memory per node
#SBATCH --gpus=1                          # Number of GPUs (for HW1, let it be 1)
#SBATCH --time=02:00:00                   # Max runtime = 2 hours (cluster limit)

#SBATCH --chdir=/homes/iws/samarjit/workspace/cse599o/cse599o-assignment1-basics  # Working directory
#SBATCH --export=ALL                      # Export environment variables
#SBATCH --output=slurm/samarjit-slurm-%j.out             # STDOUT log file (%j = JobID)
#SBATCH --error=slurm/samarjit-slurm-%j.err              # STDERR log file (%j = JobID)

# --- Example Command to run ---

srun uv run train.py