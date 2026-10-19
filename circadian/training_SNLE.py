import sys, os
import torch
import numpy as np
from sbi.inference import NLE
import pickle
import os
import argparse
import time
from torchdiffeq import odeint
from functions import  simulators_circadian, y_to_lambda_batch
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt
from sbi.analysis import pairplot

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from help_functions import ABC_rej2


def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the Priors and Simulators classes
    prior = BoxUniform(
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

    # Define the output directory
    output_dir = f"circadian/nets_depot/SNLE/{args.task}/J_{int(args.num_training/1000)}K"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    
    # Run the simulator
    theta = prior.sample((args.num_training,))
    X = simulators_circadian(theta)
    inference = NLE(prior, density_estimator="nsf")

    # train
    posterior_time = []

    start_time = time.time()
    inference.append_simulations(theta,X).train()
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    posterior = inference.build_posterior(mcmc_method="slice_np_vectorized", mcmc_parameters={"num_chains": 20, "thin": 10, "warmup_steps": 200})
    post_samples = posterior.sample((args.num_training,), x=x0)
    end_time = time.time()
    posterior_time = end_time - start_time
    
    output_file_path = os.path.join(output_dir, f"seed{args.seed}_x0{args.x0_ind}_round0.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump({'inference': inference, 'train_time': train_time, 'posterior_time': posterior_time}, f)

    pairplot(post_samples, figsize=(8,8))
    plt.savefig(f"{output_dir}/plot_x0{args.x0_ind}_seed{args.seed}_round0.png")
    plt.close()


    param_names = ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "Ka", "Kb"]
    

    # --- sequential (SNLE) ---
    proposal = prior
    num_rounds = 4
    n_per_round = args.num_training
    for r in range(num_rounds):
        theta = proposal.sample((n_per_round,))
        X = simulators_circadian(theta)
        start_time = time.time()
        inference.append_simulations(theta, X).train()   
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized", mcmc_parameters={"num_chains": 20, "thin": 10, "warmup_steps": 200})
        end_time = time.time()
        posterior_time = end_time - start_time

        output_file_path = os.path.join(output_dir, f"seed{args.seed}_x0{args.x0_ind}_round{r+1}.pkl")
        with open(output_file_path, 'wb') as f:
            pickle.dump({'inference': inference, 'train_time': train_time, 'posterior_time': posterior_time}, f)

        pairplot(post_samples, figsize=(8,8))
        plt.savefig(f"{output_dir}/plot_x0{args.x0_ind}_seed{args.seed}_round{r+1}.png")
        plt.close()
        proposal = posterior.set_default_x(x0)
        print(f"Saved inference object and elapsed time to '{output_file_path}'.")
        

    print("Training completed.")
    
def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='circadian', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--num_training', type=int, default=500_000, help='Number of simulations to run')
    parser.add_argument('--x0_ind', type=int, default=0, help='Index of the initial condition to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
