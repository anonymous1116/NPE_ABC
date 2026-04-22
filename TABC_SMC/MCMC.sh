#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=04:00:00
#SBATCH --qos=standby
#SBATCH --array=0-99               # Create a job array with indices from 1 to 10
#SBATCH --output=TABC_SMC/output_log/output_log_%A_%a.log
#SBATCH --error=TABC_SMC/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p TABC_SMC/output_log

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# Define the starting point for seed
seed_START=1
#TASK="MoG"

# Get the current N_EPOCHS value based on the job array index
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

# Run the Python script with the specified N_EPOCHS value
echo "Running with seed=$seeds"
#python TABC_SMC/TABC_MCMC2.py --task "bernoulli_glm2" --seed 1 --x0_ind 1 --num_training 1000000 --tol 1e-3 --cond_den "nsf"
python TABC_SMC/TABC_MCMC2_multi.py --task "bernoulli_glm2" --seed $seed --x0_ind $x0_ind --num_training 1000000 --tol 1e-3 --cond_den "nsf"
#python TABC_SMC/TABC_MCMC2_multi.py --task "bernoulli_glm2" --seed 1 --x0_ind 1 --num_training 1000000 --tol 1e-3 --cond_den "nsf"
echo "## Run Completed for seed=$seeds ##"
