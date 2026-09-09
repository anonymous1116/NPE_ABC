#!/bin/bash
#SBATCH --account=PES0984            # replace with your actual project account
#SBATCH --job-name=NPE_ABC
#SBATCH --output=%j.out            # %x = job name, %j = job ID
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --partition=debug
#SBATCH --time=00:30:00

module load miniconda3/24.1.2-py310
source activate BayesCalib

cd $SLURM_SUBMIT_DIR                  # run from wherever you submitted the job
python training.py