#!/bin/bash
#SBATCH --account=PES0984            # replace with your actual project account
#SBATCH --job-name=NPE_ABC
#SBATCH --output=%x_%j.out            # %x = job name, %j = job ID
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00               # adjust to how long the run actually needs
#SBATCH --partition=debug


module load miniconda3/24.1.2-py310
source activate BayesCalib

cd $SLURM_SUBMIT_DIR                  # run from wherever you submitted the job
python training.py