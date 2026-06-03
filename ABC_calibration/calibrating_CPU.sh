#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=04:00:00
#SBATCH --qos=standby
#SBATCH --array=0-99
#SBATCH --output=ABC_calibration/log_CPU/output_log_%A_%a.out
#SBATCH --error=ABC_calibration/log_CPU/error_log_%A_%a.txt

# #SBATCH --partition=a10,a100-40gb,a100-80gb

# Create the output_log directory if it doesn't exist
mkdir -p ABC_calibration/log_CPU

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
task="my_five_twomoons_err30"
num_training=3000000
tol=1e-5
# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"

#python ABC_calibration/calibrating_flow.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow_latent4.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol 
python ABC_calibration/calibrating_flow_latent5.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol
#python ABC_calibration/calibrating_flow_latent_target.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#module load conda
#conda activate /depot/wangxiao/apps/hyun18/NPE_NABC
#python ABC_calibration/calibrating_flow_latent_target.py --x0_ind 9 --seed 1 --L 1000000 --task "my_five_twomoons_err90" --num_training 5000000 --tol 1e-2 
#python ABC_calibration/calibrating_flow_experiment.py --x0_ind 1 --seed 1 --L 10000000 --task "my_five_twomoons_err90" --num_training 5000000 --tol 1e-3
