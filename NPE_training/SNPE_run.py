import sys, os
import torch
import numpy as np
from sbi.inference import NPE
import pickle
import os
import argparse
import time
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
    inference = NPE(priors, density_estimator = args.cond_den)
    proposal = priors
    c2st_results_list = []
    elapsed_time_list = []
    end_time = time.time()
        
    for _ in range(args.total_round):
        start_time = time.time()  # Start timer
        theta = proposal.sample((args.num_training,))
        x = simulators(theta)

        # In `SNLE` and `SNRE`, you should not pass the `proposal` to `.append_simulations()`.
        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        end_time = time.time()
        elapsed_time = end_time - start_time
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((10_000,), x=x0)
        c2st_results = c2st(samples, true)
        c2st_results_list.append(c2st_results[0].tolist())
        elapsed_time_list.append(elapsed_time)
        print(f"J: {int(args.num_training/1000)}K, Round: {_+1}/{args.total_round}, c2st {c2st_results}" )
        proposal = posterior.set_default_x(x0)
    
    # Define the output directory
    output_dir = f"../depot_hyun/hyun/NPE_ABC/SNPE_nets_round{args.total_round}/{args.task}/J_{int(args.num_training/1000)}K"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save the inference object using pickle in the specified directory
    # Save the inference object and elapsed time using pickle in the specified directory
    output_file_path = os.path.join(output_dir, f"{args.task}_x0{args.x0_ind}_seed{args.seed}_{args.cond_den}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time_list': elapsed_time_list, 'c2st_list': c2st_results_list}, f)
        
    print(f"Saved inference object and elapsed time to '{output_file_path}'.")

    output_dir = f"../depot_hyun/hyun/NPE_ABC/SNPE_c2st_round{args.total_round}_results/{args.task}/J_{int(args.num_training/1000)}K"   
    os.makedirs(output_dir, exist_ok=True)
    torch.save(c2st_results_list, os.path.join(output_dir, f"result_x0_{args.x0_ind}_seed_{args.seed}.pt"))  # Customize filename as needed
    
def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    parser.add_argument('--cond_den', type=str, default='nsf', help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--x0_ind', type=int, default=0, help='observation index')
    parser.add_argument('--total_round', type=int, default=0, help='observation index')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function