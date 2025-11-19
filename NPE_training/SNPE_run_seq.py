import sys, os
import torch
import numpy as np
from sbi.inference import NPE
import pickle
import os
import argparse
import time
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, Priors, observation_lists, true_Posteriors
from utils.evaluate import create_c2st_job_script
from sbibm.metrics.c2st import c2st

def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)
    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    true_posteriors = true_Posteriors(args.task)
    
    true = true_posteriors(j = args.x0_ind+1)
    
    x0 = observation_lists(args.task)[args.x0_ind]
    inference = NPE(priors)
    proposal = priors
    num_rounds = 10
    start_time = time.time()  # Start timer
    c2st_results_list = []

    # Load the inference object if it exists
    output_dir_tmp = f"../depot_hyun/hyun/NPE_ABC/SNPE_nets_{args.total_round}_temp/{args.task}/J_{int(args.num_training/1000)}K"
    
    # Create the temporal directory if it doesn't exist
    if not os.path.exists(output_dir_tmp):
        os.makedirs(output_dir_tmp)
        print(f"Temporal Directory '{output_dir_tmp}' created.")
    else:
        print(f"Temproal Directory '{output_dir_tmp}' already exists.")

    output_file_path_tmp = os.path.join(output_dir_tmp, f"{args.task}_x0{args.x0_ind}_seed{args.seed}_{args.cond_den}_tmp.pkl")

    # Check if we need to load a previous inference object
    if os.path.exists(output_file_path_tmp):
        with open(output_file_path_tmp, 'rb') as f:
            data = pickle.load(f)
            inference = data['inference']
            density_estimator = data['density_estimator']
            elapsed_time_list = data['elapsed_time_list']
            c2st_list = data['c2st_list']
            round = data['round']
            print("Loaded previous inference object.")
        round += 1
        print("round", flush = True)
        torch.manual_seed(args.seed + round)
        start_time = time.time()
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(x0)
        theta = proposal.sample((args.num_training,))
        x = simulators(theta)
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((10_000,), x=x0)
        end_time = time.time()
        elapsed_time = end_time - start_time
        c2st_results = c2st(samples, true)[0].tolist()
        print("c2st", c2st_results)
        c2st_list.append(c2st_results)
        elapsed_time_list.append(elapsed_time)
        with open(output_file_path_tmp, 'wb') as f:
            pickle.dump({'inference': inference, 'density_estimator': density_estimator, 'elapsed_time_list': [elapsed_time], 'c2st_list': [c2st_results], 'round': round}, f)
        print(f"Saved inference object and elapsed time to '{output_file_path_tmp}'.")

    else:
        round = 0
        torch.manual_seed(args.seed)
        start_time = time.time()
        proposal = priors
        theta = proposal.sample((args.num_training,))
        x = simulators(theta)
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((10_000,), x=x0)
        end_time = time.time()
        
        c2st_results = c2st(samples, true)[0].tolist()
        #proposal = posterior.set_default_x(x0)
        print("c2st", c2st_results)
        
        elapsed_time = end_time - start_time

        with open(output_file_path_tmp, 'wb') as f:
            pickle.dump({'inference': inference, 'density_estimator': density_estimator, 'elapsed_time_list': [elapsed_time], 'c2st_list': [c2st_results], 'round': round}, f)
        print(f"Saved inference object and elapsed time to '{output_file_path_tmp}'.")
        print(f"round {round} completed")
    
    if round == args.total_round-1:
        print("Training completed successfully.")
        output_dir = f"../depot_hyun/hyun/NPE_ABC/SNPE_nets_seq_round{args.total_round}/{args.task}/J_{int(args.num_training/1000)}K"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Save Directory '{output_dir}' created.")
        else:
            print(f"Save Directory '{output_dir}' already exists.")

        output_file_path = os.path.join(output_dir, f"{args.task}_x0{args.x0_ind}_seed{args.seed}_{args.cond_den}.pkl")
        with open(output_file_path, 'wb') as f:
            pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time_list': elapsed_time, 'c2st_list': c2st_results_list}, f)
        
        # dump the residuals
        os.remove(output_file_path_tmp)
        
    
    else:
        print(f"Training not completed. Generating a new job script..., round {round}/{args.total_round-1}")
        create_job_script(args)

    
def create_job_script(args):
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --account=statdept
#SBATCH --time=04:00:00
#SBATCH --qos=standby
#SBATCH --output=SNPE/output_log/output_log_%A_%a.log
#SBATCH --error=SNPE/output_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p SNPE/output_log

# Load the required Python environment
module load conda
conda activate /depot/wangxiao/apps/hyun18/NPE_NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NPE_ABC
cd $SLURM_SUBMIT_DIR

# Run the Python script for additional training
echo "Resuming training..."
python NPE_training/SNPE_run_seq.py --task "{args.task}" --seed {args.seed} --cond_den "{args.cond_den}" --num_training {args.num_training} --x0_ind {args.x0_ind}
echo "## Job completed for task {args.task} with seed {args.seed} ##"
"""
    # Create the directory if it doesn't exist
    output_dir = f"SNPE/{args.task}/J_{int(args.num_training/1000)}K/slurm_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    job_file_path = os.path.join(output_dir, f"{args.cond_den}_{args.x0_ind}_{args.seed}_round{args.total_round}_continued.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created and submitted: {job_file_path}")

    # Submit the job immediately
    os.system(f"sbatch {job_file_path}")
    print(f"Job {job_file_path} submitted.")


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, required=True, help='Simulation type: twomoons, SLCP, or MoG')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--cond_den', type=str, required=True, help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    parser.add_argument('--x0_ind', type=int, default=0, help='observation index')
    parser.add_argument('--total_round', type=int, default=2, help='total round of SNPE')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
