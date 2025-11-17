import os
import torch
import numpy as np
import subprocess
import argparse
import sys
# Add the parent directory to the system path to import simulator.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import observation_lists

def create_c2st_job_script(args, x0_ind, seed, post_n_samples, use_gpu=False):
    sbatch_gpu_options = """
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=a10,a30
#SBATCH --mem=80G
""" if use_gpu else """
#SBATCH -p cpu
"""

    sbatch_activate_options = """
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC
""" if use_gpu else """
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC
"""

    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --account=statdept
#SBATCH -q standby
{sbatch_gpu_options}
#SBATCH --output={args.measure}/{args.task}/output_log/output_log_%A.log
#SBATCH --error={args.measure}/{args.task}/error_log/error_log_%A.txt

mkdir -p {args.measure}/{args.task}/output_log
mkdir -p {args.measure}/{args.task}/error_log

# Load the required Python environment
module load conda
{sbatch_activate_options}

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

# Run the Python script for the current simulation
echo "Running simulation for task '{args.task}', '{args.num_training}' '{args.methods}', x0_ind={j}, seed={seed}..."
python ./utils/get_measure.py --task {args.task} --num_training {args.num_training}  --x0_ind {x0_ind} --seed {seed} --post_n_samples {post_n_samples} 
echo "## Job completed for task '{args.task}', '{args.methods}', x0_ind={x0_ind}, seed={seed}" ##"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"NPE_ABC/{args.measure}/{args.task}/slurm_files"
    os.makedirs(output_dir, exist_ok=True)
    job_file_path = os.path.join(output_dir, f"{args.task}_NPE_{int(args.num_training/1000)}K_c2st_x0_ind{x0_ind}_seed{seed}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")


def main(args):
    x0_list = observation_lists
    seeds = np.arange(1, 11)  # Use np.arange instead of np.range
    # Create SLURM job scripts for each combination of x0_list and 10 runs
    gpu_ind = True if torch.cuda.is_available() else False

    for i in range(len(x0_list.tolist())):
        for j in seeds:
            create_c2st_job_script(args, x0_ind = i, seed = j, use_gpu = gpu_ind)
