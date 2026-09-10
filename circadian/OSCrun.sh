#!/bin/bash
#SBATCH --account=PES0984            # replace with your actual project account
#SBATCH --job-name=NPE_ABC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=09:30:00
#SBATCH --output=circadian/output_log/%j.out            # %x = job name, %j = job ID
#SBATCH --error=circadian/output_log/%j.err

mkdir -p circadian/output_log

SLURM_SUBMIT_DIR=/users/PES0984/hhyun116/NPE_ABC

module load miniconda3/24.1.2-py310
source activate BayesCalib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd $SLURM_SUBMIT_DIR                  # run from wherever you submitted the job
python circadian/training.py --task "circadian" --method NPE --num_training 100000 --seed 2