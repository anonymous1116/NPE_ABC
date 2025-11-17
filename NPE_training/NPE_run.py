import sys
import torch
import numpy as np
from sbi.inference import NPE
import pickle
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, Priors, observation_lists, Posteriors, Bounds
from sbibm.metrics.c2st import c2st
import subprocess

def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    simulators = Simulators(args.task)

    # Sample theta from the prior
    theta = priors.sample((args.num_training,))

    # Run the simulator
    X = simulators(theta)

    # Create inference object
    inference = NPE(prior=priors, density_estimator=args.cond_den)
    inference = inference.append_simulations(theta, X)

    # Train the density estimator and build the posterior
    print(f"training_start")
    start_time = time.time()  # Start timer
    density_estimator = inference.train()
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Training with {args.cond_den}")

    # Define the output directory
    output_dir = f"../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save the inference object using pickle in the specified directory
    # Save the inference object and elapsed time using pickle in the specified directory
    output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}_{args.cond_den}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time': elapsed_time}, f)
    
    print(f"Saved inference object and elapsed time to '{output_file_path}'.")

def create_c2st_job_script(args, j, i, post_n_samples, use_gpu=False):
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
echo "Running simulation for task '{args.task}', '{args.num_training}' '{args.methods}', {args.measure} j={j}, i={i}..."
python ./utils/get_measure.py --task {args.task} --num_training {args.num_training} --measure {args.measure} --x0_ind {j} --seed {i} --post_n_samples {post_n_samples} 
echo "## Job completed for task '{args.task}', '{args.methods}', {args.measure} j={j}, i={i}" ##"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"NPE_ABC/{args.measure}/{args.task}/slurm_files"
    os.makedirs(output_dir, exist_ok=True)
    job_file_path = os.path.join(output_dir, f"{args.task}_NPE_{int(args.num_training/1000)}K_c2st_j{j}_i{i}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")



def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function

    #task_params = get_task_parameters(args.task)
    limits = Bounds(args.task)
    x0 = observation_lists[args.task]
    gpu_ind = True if torch.cuda.is_available() else False
#
    for i in range(len(x0)):
        create_c2st_job_script(args.task, args.num_training, "c2st", i = i, j = args.seed, use_gpu = gpu_ind)
    