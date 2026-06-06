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
from utils.evaluate import create_c2st_job_script
from sbibm.metrics.c2st import c2st

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
    
    # Define the output directory
    output_dir = f"../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/{args.task}/J_{int(args.num_training/1000)}K"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save the inference object using pickle in the specified directory
    # Save the inference object and elapsed time using pickle in the specified directory
    #output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}.pkl")
    #with open(output_file_path, 'wb') as f:
    #    pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time': elapsed_time}, f)
    
    #print(f"Saved inference object and elapsed time to '{output_file_path}'.")

    torch.save(elapsed_time, f"{output_dir}/{args.task}_{args.seed}_time.pt")

    x0_list = observation_lists(args.task)
    limits = Bounds(args.task)
    posterior = true_Posteriors(args.task)
    
    for j in range(len(x0_list)):
        x_o = x0_list[j]
        x_o = torch.tensor(x_o, dtype = torch.float32)

        posterior_NPSE = inference.build_posterior().set_default_x(x_o)
        time0 = time.time()
        sample_post = posterior_NPSE.sample((10000,))
        time1 = time.time()

        elapsed_time =time1-time0
        # Get true posterior      
        if args.task in task_benchmark:
            true_sample = posterior(j = j+1)
        else:
            true_sample = posterior(torch.tensor(x_o), n_samples=10_000, bounds=limits)
        
        measure = "c2st"

        dist = c2st(true_sample, sample_post)
        print(f"{measure}: {dist}")
        output_dir = f"../depot_hyun/hyun/NPE_ABC/NPSE_{measure}_NABC_results/{args.task}/J_{int(args.num_training/1000)}K"   
        
        os.makedirs(output_dir, exist_ok=True)
        torch.save(dist, os.path.join(output_dir, f"result_x0_{j}_seed_{args.seed}.pt"))  # Customize filename as needed
        torch.save(elapsed_time, os.path.join(output_dir, f"result_x0_{j}_seed_{args.seed}_time.pt"))  # Customize filename as needed
        
    print(f"Saved inference object and elapsed time to '{output_dir}/{args.task}_{args.seed}'.",flush =True)


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