#!/bin/bash
#SBATCH --account=PES0984            # replace with your actual project account
#SBATCH --job-name=TABC_dist
#SBATCH --clusters=ascend          # or: cardinal, pitzer, ascend
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-10:00:00
#SBATCH --output=NPE_ABC/output_log/%j.out            # %x = job name, %j = job ID
#SBATCH --error=NPE_ABC/output_log/%j.err
#SBATCH --array=1               # Create a job array with indices from 1 to 10
###SBATCH --gpus-per-node=1


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
num_training=1000000
tol=1e-5
#cdim=10
# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"

#python TABC_dist/target.py  --etaprime 1e-2 --x0_ind $x0_ind --seed $seed --task $task --num_training $num_training --tol $tol --L $L


#python TABC_dist/target.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol --cdim $cdim

echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#module load conda
#conda activate /depot/wangxiao/apps/hyun18/NPE_NABC
#python ABC_calibration/calibrating_flow_latent5.py --x0_ind 1 --seed 1 --L 1000000 --task "slcp" --num_training 3000000 --tol 1e-2 
#python ABC_calibration/calibrating_flow_experiment.py --x0_ind 1 --seed 1 --L 1000000 --task "my_five_twomoons_err40" --num_training 3000000 --tol 1e-2
#python DP/calibrating_DP.py --x0_ind 1 --seed 1 --L 10000000 --task "table_dp_77" --num_training 3000000 --tol 1e-3
#python ABC_calibration/SE_calib.py --x0_ind 1 --seed 4 --L 10000000 --task "mog_10" --num_training 300000 --tol 1e-3 --SDE "vp"

python TABC_dist/target.py --x0_ind 0 --seed 1 --L 1000000 --task "bernoulli_glm2_disperse20" --num_training 100000 --tol 1e-2