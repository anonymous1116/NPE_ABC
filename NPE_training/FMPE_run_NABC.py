import sys, os
import torch
import numpy as np
from sbi.inference import FMPE
import pickle
import os
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, Priors, observation_lists, Bounds
import subprocess


def create_c2st_job_script(task, num_training, measure, x0_ind, seed, post_n_samples, cond_den, method = "NPE", use_gpu=False, embed=False,cdim=None):
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
    if embed:
        implement_options = "get_measure_embed"
        cdim_arg = f"--cdim {cdim}" if cdim is not None else ""
        method_arg = f"" 
    else:
        implement_options = "get_measure"
        cdim_arg = ""
        method_arg = f"--method {method}"
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:59:00
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

python ./utils/{implement_options}_NABC.py --task {task} --num_training {num_training} --measure {measure} --x0_ind {x0_ind} --seed {seed} --post_n_samples {post_n_samples} --cond_den {cond_den} {cdim_arg} {method_arg}
echo "## Job completed for task '{task}', x0_ind={x0_ind}, seed={seed}" ##"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"NPE_ABC/{measure}/{task}/slurm_files"
    os.makedirs(output_dir, exist_ok=True)
    job_file_path = os.path.join(output_dir, f"{task}_{method}_{int(num_training/1000)}K_c2st_x0_ind{x0_ind}_seed{seed}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")




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
    inference = FMPE(prior=priors)
    inference = inference.append_simulations(theta, X)

    # Train the density estimator and build the posterior
    print(f"training_start")
    start_time = time.time()  # Start timer
    density_estimator = inference.train()
    end_time = time.time()  # End timer

    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Define the output directory
    output_dir = f"../depot_hyun/hyun/NPE_ABC/FMPE_nets_NABC/{args.task}/J_{int(args.num_training/1000)}K"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save the inference object using pickle in the specified directory
    # Save the inference object and elapsed time using pickle in the specified directory
    output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time': elapsed_time}, f)
    
    print(f"Saved inference object and elapsed time to '{output_file_path}'.")


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
    x0_list = observation_lists(args.task)
    gpu_ind = True if torch.cuda.is_available() else False

    for i in range(len(x0_list.tolist())):
        create_c2st_job_script(task = args.task, 
                               num_training = args.num_training, 
                               measure = "c2st", 
                               x0_ind = i, 
                               seed = args.seed, 
                               post_n_samples =10_000, 
                               cond_den = "nothing",
                               method = "FMPE", 
                               use_gpu = gpu_ind,
                               embed = False)
        