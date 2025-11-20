#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=1-04:00:00
#SBATCH --qos=normal
#SBATCH --array=0-99               # Create a job array with indices from 1 to 10
#SBATCH --output=SNPE/NPE_nsf/output_log/output_log_%A_%a.log
#SBATCH --error=SNPE/NPE_nsf/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p NPE/NPE_nsf/output_log

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 
num_training=100000
task="two_moons"
total_round=5
echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"
python NPE_training/SNPE_run.py --task $task --seed $seed --x0_ind $x0_ind --num_training $num_training --cond_den "nsf" --total_round $total_round
#python NPE_training/SNPE_run_seq.py --task "two_moons" --seed $seed --x0_ind $x0_ind --num_training $num_training --cond_den "nsf" --total_round $total_round

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"


#python NPE_training/SNPE_run_seq.py --task "two_moons" --seed 1 --x0_ind 1 --num_training 1000 --cond_den "nsf" --total_round 5
#python NPE_training/SNPE_run.py --task "two_moons" --seed 1 --x0_ind 1 --num_training 1000 --cond_den "nsf" --total_round 2

