import sys, os
import torch
import numpy as np
from sbi.inference import NPE, FMPE, NPSE
import pickle
import os
import argparse
import time
from torchdiffeq import odeint
from functions import  ode_model, make_fourier_design, y_to_lambda_batch
from sbi.utils import BoxUniform

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
#from utils.evaluate import create_c2st_job_script

def simulators_circadian(theta, device = "cpu", max_ODEtime = 500, T_field = 66):
    """
    Simulates the circadian model for a batch of parameter sets theta.
    Args:
        theta: (batch, 12) tensor of parameters
    """

    # --- Settings ---
    M_obs_time = np.arange(max_ODEtime - T_field, max_ODEtime)  # (max_ODEtime-T_field+1):max_ODEtime

    # --- Initial conditions, replicated per batch item ---
    batch_size = theta.size(0)
    y0 = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)  # M, P, Pp all start at 0

    # --- Time points ---
    t_eval = torch.arange(1, max_ODEtime + 1, dtype=torch.float64, device=device)

    # --- Solve all trajectories at once ---
    # odeint's func signature is func(t, y) -> dy/dt; wrap theta_batch via closure

    theta = torch.column_stack([torch.ones(batch_size) * 24.44, 
                                theta, 
                                torch.ones(batch_size) * 8.0, 
                                torch.ones(batch_size) * 4.0])  
    
    sol = odeint(
        lambda t, y: ode_model(t, y, theta),
        y0,
        t_eval,
        method="dopri5",   # Dormand-Prince, same family as R's ode45
        rtol=1e-10,
        atol=1e-10,
    )
    # sol shape: (time, batch, 3)  ->  matches deSolve output per-batch-item if you index sol[:, i, :]

    ModelRun = sol.permute(1, 0, 2)  # (batch, time, 3) if you prefer batch-first
    M_batch = ModelRun[:, M_obs_time, 0]          # (batch, T_y), the M trajectories only
    return y_to_lambda_batch(M_batch, deg=15).cpu()  # (batch, n_freqs), the S_sq per frequency

def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the Priors and Simulators classes
    priors = BoxUniform(low = torch.ones(9)*1e-6, high = torch.tensor([0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0]))
    
    # Sample theta from the prior
    theta = priors.sample((args.num_training,))
    theta = theta.float()   # or theta.to(torch.float32)
    
    # Run the simulator
    X = simulators_circadian(theta, device = device)
    X = X.float()            # or X.to(torch.float32)
        
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
    output_dir = f"nets_depot/{args.method}/{args.task}/J_{int(args.num_training/1000)}K"
    
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
