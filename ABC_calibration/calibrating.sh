#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:39:00
#SBATCH --account=statdept
#SBATCH --gpus-per-node=1
#SBATCH --mem=170G
#SBATCH --qos=normal
#SBATCH --partition=a10
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

L=1000000000
task="mog_10"
num_training=3000000 
tol=1e-5

# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"

#python ABC_calibration/calibrating_flow.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol

python ABC_calibration/calibrating_flow_experiment.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow_linear_embed.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow_linear_embed.py --x0_ind 1 --seed 1 --L 10000000 --task "my_five_twomoons_err10" --num_training 1000000 --tol 1e-3
#python ABC_calibration/calibrating_flow_experiment2.py --x0_ind $x0_ind --seed $seed --task $task --num_training 3000000 
#python ABC_calibration/calibrating_flow_experiment2.py --x0_ind 1 --seed 1 --task "my_five_twomoons" --num_training 3000000 

#python ABC_calibration/calibrating_flow.py --x0_ind 1 --seed 1 --L 10000000 --task "my_five_twomoons_err2" --num_training 100000 --tol 1e-3
#python ABC_calibration/calibrating_flow_experiment.py --x0_ind 1 --seed 1 --L 10000000 --task "my_five_twomoons_err2" --num_training 1000000 --tol 1e-3

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

