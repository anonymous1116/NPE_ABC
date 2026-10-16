#!/bin/bash
#SBATCH --account=PES0984
#SBATCH --job-name=sensi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=circadian/output_log/%x_%A_%a.out
#SBATCH --error=circadian/output_log/%x_%A_%a.err
#SBATCH --array=450-899           # 45 * N_CHUNKS tasks, at most 50 running at once

# N_CHUNKS = (rows in Techno_thin.txt) / 1000
N_CHUNKS=20

module load miniconda3/24.1.2-py310
source activate BayesCalib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

SLURM_SUBMIT_DIR=/users/PES0984/hhyun116/NPE_ABC
cd $SLURM_SUBMIT_DIR

# decode the array index into (chunk, scale factor, parameter)
amounts=(0.6 0.8 1.0 1.2 1.4)
id=$SLURM_ARRAY_TASK_ID
chunk=$(( id % N_CHUNKS ))
rest=$(( id / N_CHUNKS ))
amount=${amounts[$(( rest % 5 ))]}
theta_index=$(( rest / 5 + 1 ))     # 1..9

echo "task=$id start=$chunk theta_index=$theta_index sensi_amount=$amount"

python circadian/sensitivity.py --method MCMC --start $chunk \
    --theta_index $theta_index --sensi_amount $amount --maxtime 10000

#python circadian/sensitivity.py --method MCMC --start 0 --theta_index 4 --sensi_amount 0.6 --maxtime 520