#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=170G
#SBATCH --qos=standby
#SBATCH --partition=a30
#SBATCH --array=0-99
#SBATCH --output=ABC_calibration/log/output_log_%A_%a.out
#SBATCH --error=ABC_calibration/log/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p ABC_calibration/output_log

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
task="two_moons"
num_training=10000
tol=1e-2

# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"

python ABC_calibration/calibrating.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#python benchmark/benchmark_calibrating.py --x0_ind 1 --seed 1 --L 100000000 --task "slcp" --num_training 300000 --tol 1e-4
#python ABC_calibration/calibrating.py --x0_ind 1 --seed 1 --task "two_moons" --L 1000000 --num_training 10000 --tol 1e-2 