import os
import torch
import numpy as np
import subprocess
import argparse
import sys
# Add the parent directory to the system path to import simulator.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import observation_lists

def create_c2st_job_script(task, num_training, measure, x0_ind, seed, post_n_samples, cond_den, use_gpu=False):
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
#SBATCH --output=NPE_ABC/{measure}/{task}/output_log/output_log_%A.log
#SBATCH --error=NPE_ABC/{measure}/{task}/error_log/error_log_%A.txt

mkdir -p NPE_ABC/{measure}/{task}/output_log
mkdir -p NPE_ABC/{measure}/{task}/error_log

# Load the required Python environment
module load conda
{sbatch_activate_options}

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

# Run the Python script for the current simulation
echo "Running simulation for task '{task}', '{num_training}', x0_ind={x0_ind}, seed={seed}..."
python ./utils/get_measure.py --task {task} --num_training {num_training} --measure {measure} --x0_ind {x0_ind} --seed {seed} --post_n_samples {post_n_samples} --cond_den {cond_den} 
echo "## Job completed for task '{task}', x0_ind={x0_ind}, seed={seed}" ##"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"NPE_ABC/{measure}/{task}/slurm_files"
    os.makedirs(output_dir, exist_ok=True)
    job_file_path = os.path.join(output_dir, f"{task}_NPE_{int(num_training/1000)}K_c2st_x0_ind{x0_ind}_seed{seed}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")



def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--methods', type=str, default='NABC', help='methods type: NABC, NPE_maf, NPE_nsf')
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl or slcp')
    parser.add_argument('--measure', type=str, default='c2st', help='Simulation type: c2st, SW')
    parser.add_argument('--post_n_samples', type=int, default=10_000, help='Number of samples from posterior distributions')
    parser.add_argument("--num_training", type=int, default=500_000,
                        help="Number of simulations for training (default: 500_000)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    return parser.parse_args()


def main(args):
    print("asdf")
    x0_list = observation_lists
    seeds = np.arange(1, 11)  # Use np.arange instead of np.range
    # Create SLURM job scripts for each combination of x0_list and 10 runs
    gpu_ind = True if torch.cuda.is_available() else False

    print("distribute")
    for i in range(len(x0_list.tolist())):
        for j in seeds:
            create_c2st_job_script(args.task, args.num_traning, args.measure, x0_ind = i, seed = j, cond_den = args.cond_den, use_gpu = gpu_ind)
    print("distribute end")
    

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function


