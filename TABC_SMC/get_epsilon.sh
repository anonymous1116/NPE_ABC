#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=02:00:00
#SBATCH --qos=standby
#SBATCH --array=0-99               # Create a job array with indices from 1 to 10
#SBATCH --output=TABC_SMC/get_epsilon_output_log/output_log_%A_%a.log
#SBATCH --error=TABC_SMC/get_epsilon_output_log/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p TABC_SMC/get_epsilon_output_log

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
#L=100000000

L=1000000
task="bernoulli_glm2"
num_training=10000000 
tol=1e-3

# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"


python TABC_SMC/get_epsilon_by_calibration.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python TABC_SMC/get_epsilon_by_calibration.py --x0_ind 1 --seed 1 --L 10000000 --task "bernoulli_glm2" --num_training 1000000  --tol 1e-3

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

