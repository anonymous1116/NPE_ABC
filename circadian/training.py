import sys, os
import torch
import numpy as np
from sbi.inference import NPE, FMPE, NPSE
import pickle
import os
import argparse
import time
from torchdiffeq import odeint
from functions import  simulators_circadian
from sbi.utils import BoxUniform
#from help_functions import ABC_rej2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
#from utils.evaluate import create_c2st_job_script

def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the Priors and Simulators classes
    priors = BoxUniform(
        low=torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2]),  # k1..k7 stay tiny-safe, Ka/Kb raised
        high=torch.tensor([0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0]),
    )
    
    # Sample theta from the prior
    theta = priors.sample((args.num_training,))
    theta = theta.float()   # or theta.to(torch.float32)
    theta = theta.to(device)

    # Run the simulator
    X = simulators_circadian(theta, device = device)
    start_time = time.time()
    X = X.float()            # or X.to(torch.float32)
    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"Simulation completed in {simulation_time:.2f} seconds")



    X_np = X.cpu().numpy()  # move off GPU, convert to numpy for plotting
    n_freqs = X_np.shape[1]

    fig, axes = plt.subplots(1, n_freqs, figsize=(4 * n_freqs, 4))
    if n_freqs == 1:
        axes = [axes]  # keep iterable if there's only one column

    for i, ax in enumerate(axes):
        ax.hist(X_np[:, i], bins=50)
        ax.set_title(f"S_sq[{i}] (frequency {i+1})")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    plt.tight_layout()
    plt.savefig("circadian/X_distributions.png", dpi=150)
    plt.show()

    if 0:
        print(torch.max(X, dim=0).values, torch.min(X, dim=0).values)
        
        # Create inference object
        if args.method == "FMPE":
            inference = FMPE(prior=priors)
        elif args.method == "NPSE":
            inference = NPSE(prior = priors)
        elif args.method == "NPSE_vp":
            inference = NPSE(prior = priors, sde_type="vp")
        else:
            inference = NPE(prior=priors, density_estimator="nsf")
        inference = inference.append_simulations(theta, X)

        # Train the density estimator and build the posterior
        print(f"training_start")
        start_time = time.time()  # Start time
        density_estimator = inference.train()
        end_time = time.time()  # End timer

        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Define the output directory
        output_dir = f"circadian/nets_depot/{args.method}/{args.task}/J_{int(args.num_training/1000)}K"
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")

        # Save the inference object using pickle in the specified directory
        # Save the inference object and elapsed time using pickle in the specified directory
        output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}.pkl")
        
        if args.method in ["NPSE", "NPSE_vp"]:
            torch.save({
                'net_state_dict': density_estimator.net.state_dict(),
                'embedding_net_state_dict': density_estimator._embedding_net.state_dict(),
                'full_state_dict': density_estimator.state_dict(),  # includes all buffers
                'elapsed_time': elapsed_time
            }, output_file_path.replace('.pkl', '.pt'))
        else:
            with open(output_file_path, 'wb') as f:
                pickle.dump({'density_estimator': density_estimator, 'posterior': inference.build_posterior(density_estimator), 'elapsed_time': elapsed_time}, f)
            
        print(f"Saved inference object and elapsed time to '{output_file_path}'.")

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='circadian', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    parser.add_argument('--method', type=str, default='NPE', help='Method type: NPE, FMPE, NPSE')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
