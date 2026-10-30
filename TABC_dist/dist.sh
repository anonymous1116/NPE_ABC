#!/bin/bash
#SBATCH --account=PES0984
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen #debug or preemptible
#SBATCH --job-name=TABC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00            # the debug limit is 1 hour
#SBATCH --output=slurm/TABC_dist/output_log/%j.log
#SBATCH --error=slurm/TABC_dist/output_log/%j.err
#SBATCH --gpus-per-node=1
##SBATCH --array=0-99               # Create a job array with indices from 0 to 99

mkdir -p slurm/TABC_dist/output_log


SLURM_SUBMIT_DIR=/users/PES0984/hhyun116/NPE_ABC

module load miniconda3/24.1.2-py310
source activate BayesCalib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


# Calculate seed and dim_out
seed=$((SLURM_ARRAY_TASK_ID / 10 + 1))
#L=100000000

L=1000000000
task="bernoulli_glm2_disperse60"
num_training=500000
tol=1e-5
#cdim=10
# Run the calibrate_amor.py
x0_ind=$((SLURM_ARRAY_TASK_ID % 10)) 

echo "[$(date)] Starting job: x0_ind=$x0_ind, seed=$seed, L=$L"
python TABC_dist/Tdist.py --x0_ind $x0_ind --seed $seed --L $L --task $task --num_training $num_training --tol $tol 
echo "[$(date)] Job complete: x0_ind=$x0_ind, seed=$seed"

#module load conda
#conda activate /depot/wangxiao/apps/hyun18/NPE_NABC
#python ABC_calibration/calibrating_flow_latent5.py --x0_ind 1 --seed 1 --L 1000000 --task "slcp" --num_training 3000000 --tol 1e-2 
#python ABC_calibration/calibrating_flow_experiment.py --x0_ind 1 --seed 1 --L 1000000 --task "my_five_twomoons_err40" --num_training 3000000 --tol 1e-2
#python DP/calibrating_DP.py --x0_ind 1 --seed 1 --L 10000000 --task "table_dp_77" --num_training 3000000 --tol 1e-3
#python ABC_calibration/SE_calib.py --x0_ind 1 --seed 4 --L 10000000 --task "mog_10" --num_training 300000 --tol 1e-3 --SDE "vp"

#python TABC_dist/Tdist.py --x0_ind 0 --seed 1 --L 100000000 --task "bernoulli_glm2_disperse20" --num_training 500000 --tol 1e-4