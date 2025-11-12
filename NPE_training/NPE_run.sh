#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=04:00:00
#SBATCH --qos=standby
#SBATCH --array=1-10               # Create a job array with indices from 1 to 10
#SBATCH --output=NPE/NPE_nsf/output_log/output_log_%A_%a.log
#SBATCH --error=NPE/NPE_nsf/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p NPE/NPE_nsf/output_log

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
seeds=$((seed_START + SLURM_ARRAY_TASK_ID - 1))

# Run the Python script with the specified N_EPOCHS value
echo "Running with seed=$seeds"
python NPE_training/NPE_run.py --task "my_twomoons" --seed $seeds --cond_den "nsf" --num_training 50000
#python NPE_training/NPE_run.py --task "my_twomoons" --seed 1 --cond_den "nsf" --num_training 10000
echo "## Run Completed for seed=$seeds ##"