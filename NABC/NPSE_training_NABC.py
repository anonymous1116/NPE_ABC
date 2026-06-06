import sys, os
import torch
import numpy as np
from sbi.inference import NPSE
import pickle
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, Priors, observation_lists, Bounds, true_Posteriors, task_benchmark
from sbibm.metrics.c2st import c2st
import subprocess


def submit_eval_job(task, seed, num_training, obs_idx):
    job_name = f"eval_{task}_x0{obs_idx}_s{seed}"
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/logs/{job_name}_%j.out
#SBATCH --error=../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/logs/{job_name}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=00:59:00
#SBATCH --qos=standby

# Create the output_log directory if it doesn't exist
mkdir -p ../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/logs

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

python NABC/eval_NPSE.py --task {task} --seed {seed} --num_training {num_training} --obs_idx {obs_idx}
"""
    slurm_dir = f"../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/slurm_scripts"
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_path = os.path.join(slurm_dir, f"{job_name}.sh")

    with open(slurm_path, 'w') as f:
        f.write(script)

    result = subprocess.run(['sbatch', slurm_path], capture_output=True, text=True)
    print(f"Submitted {job_name}: {result.stdout.strip()}", flush=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", flush=True)

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
    inference = NPSE(prior=priors)
    inference = inference.append_simulations(theta, X)

    # Train the density estimator and build the posterior
    print(f"training_start")
    start_time = time.time()  # Start timer
    inference.train()
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Save timing
    nets_dir = f"../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/{args.task}/J_{int(args.num_training/1000)}K"
    os.makedirs(nets_dir, exist_ok=True)
    torch.save(elapsed_time, os.path.join(nets_dir, f"{args.task}_{args.seed}_time.pt"))

    # Generate and save samples for each x0
    x0_list = observation_lists(args.task)
    samples_dir = f"../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/NPSE_samples_NABC/{args.task}/J_{int(args.num_training/1000)}K"
    os.makedirs(samples_dir, exist_ok=True)

    for j, x_o in enumerate(x0_list):
        x_o = torch.tensor(x_o, dtype=torch.float32)
        posterior_NPSE = inference.build_posterior().set_default_x(x_o)

        t0 = time.time()
        sample_post = posterior_NPSE.sample((10000,))
        elapsed_sample = time.time() - t0

        torch.save(sample_post, os.path.join(samples_dir, f"samples_x0_{j}_seed_{args.seed}.pt"))
        torch.save(elapsed_sample, os.path.join(samples_dir, f"samples_x0_{j}_seed_{args.seed}_time.pt"))
        print(f"Saved samples for x0={j}", flush=True)

        # Submit eval job for this x0
        submit_eval_job(args.task, args.seed, args.num_training, obs_idx=j)
def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--num_training', type=int, default=100_000, help='Number of simulations to run')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function    