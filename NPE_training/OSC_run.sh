#!/bin/bash
#SBATCH --account=PES0984            # replace with your actual project account
#SBATCH --job-name=NPE_ABC
#SBATCH --clusters=cardinal          # or: cardinal, pitzer, ascend
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-10:00:00
#SBATCH --output=NPE_ABC/output_log/%j.out            # %x = job name, %j = job ID
#SBATCH --error=NPE_ABC/output_log/%j.err
#SBATCH --array=1-10               # Create a job array with indices from 1 to 10
###SBATCH --gpus-per-node=1

mkdir -p circadian/output_log

SLURM_SUBMIT_DIR=/users/PES0984/hhyun116/NPE_ABC

module load miniconda3/24.1.2-py310
source activate BayesCalib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

seeds=$SLURM_ARRAY_TASK_ID

cd $SLURM_SUBMIT_DIR                  # run from wherever you submitted the job
#python circadian/training.py --task "circadian" --method NPE --num_training 500000 --seed $seeds --x0_ind 2
python NPE_training/NPE_run.py --task "bernoulli_glm2_disperse40" --num_training 1000000 --seed $seeds