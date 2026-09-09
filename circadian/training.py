import sys, os
import torch
import numpy as np
from sbi.inference import NPE, FMPE, NPSE
import pickle
import os
import argparse
import time
from torchdiffeq import odeint
from functions import  simulators_circadian, y_to_lambda_batch
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from help_functions import ABC_rej2

#from utils.evaluate import create_c2st_job_script


def filter_bottom_99(theta, X):
    """
    Keeps only rows where every column of X is below its own 99th percentile.
    Returns filtered theta and X, aligned by row.
    """
    # Compute the 99th percentile threshold per column
    thresholds = torch.quantile(X, 0.99, dim=0)  # shape: (n_freqs,)

    # A row is kept only if ALL its columns are below their respective threshold
    mask = (X <= thresholds).all(dim=1)  # shape: (batch,)

    theta_filtered = theta[mask]
    X_filtered = X[mask]

    print(f"Kept {mask.sum().item()} / {X.shape[0]} samples "
          f"({100 * mask.float().mean().item():.1f}%)")

    return theta_filtered, X_filtered


def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the Priors and Simulators classes
    priors = BoxUniform(
        low=torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1, 1e-1]),  # k1..k7 stay tiny-safe, Ka/Kb raised
        high=torch.tensor([0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0]),
    )

    # observation
    y_mat = np.loadtxt("./circadian/y_mat.txt")
    x0s = y_to_lambda_batch(torch.tensor(y_mat))
    x0s = x0s.float()
    x0 = x0s[args.x0_ind]
    if x0.ndim == 1:
        x0= torch.reshape(x0, (1, x0.size(0)))

    # Run the simulator
    start_time = time.time()
    batch_size = 500_000
    num_chunks = args.num_training * 100 // batch_size

    X_abc, Y_abc = [], []
    
    for i in range(num_chunks + 1): 
        start = i * batch_size
        end = (i + 1) * batch_size if (i + 1) * batch_size < args.num_training else args.num_training
        nums = end-start

        Y_chunk = priors.sample((nums,))
        Y_chunk = Y_chunk.float()   # or Y_chunk.to(torch.float32)
        Y_chunk = Y_chunk.to(device)
        
        X_chunk = simulators_circadian(Y_chunk, device =device)
        
        index_ABC = ABC_rej2(x0, X_chunk, .01, device)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"Simulation completed in {simulation_time:.2f} seconds")        

    #theta, X = filter_bottom_99(theta, X)
    #print(f"After filtering, theta shape: {theta.shape}, X shape: {X.shape}")

    X_np = X_abc.cpu().numpy()  # move off GPU, convert to numpy for plotting
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
    parser.add_argument('--x0_ind', type=int, default=0, help='Index of the initial condition to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
