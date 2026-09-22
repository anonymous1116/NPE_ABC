#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=2-04:00:00
#SBATCH --qos=normal
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
#python NPE_training/NPE_run_embed.py --task "slcp_distractors" --seed $seeds --cond_den "nsf" --num_training 1000000 --cdim 5
#python NPE_training/NPSE_run.py --task "bernoulli_glm2" --seed $seeds --num_training 1500000
#python NPE_training/NPE_DP_run.py --task "table_dp_66" --seed $seeds --num_training 500000 --method "FMPE"
python NPE_training/NPE_run.py --task "fold5" --seed $seeds --num_training 500000 --method "NPE"
#python NPE_training/SNPE_run.py --task "double_slcp_summary_transform2" --seed 1 --num_training 1000 --cond_den "nsf" --x0_ind 0
#python NPE_training/NPE_DP_run.py --task "table_dp_77" --seed 1 --num_training 1000 --method "NPE"

echo "## Run Completed for seed=$seeds ##"
