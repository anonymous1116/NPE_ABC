import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, Posteriors
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2, forward_from_Z_chunked, covs_chunked
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot

def main(args):
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    NABC_results = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    priors = Priors(args.task)
    true_posteriors = Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    #Y_cal = priors.sample((1_000_000,))
    #X_cal = simulators(Y_cal)
        
    # Initialize the Priors and Simulators classes and ABC_methods
    if args.task in ["two_moons"]:
        tol0 = 1e-2
    else:
        tol0 = 1e-3

    if L > 1_000_000_000 + 1:
        tol0 = 1e-5
    
    chunk_size = 50_000_000
    num_chunks = L // chunk_size
    
    start_time = time.time()
    x0 = observation_lists(args.task)[args.x0_ind]

    print(x0)
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1,x0.size(0)))
        
    chunk_size_cal = 10_000
    print("x0_size", x0.size(), flush = True)
    #print("X_cal size", X_cal.size(), flush = True)
    
    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    posterior = saved_data["posterior"]
    adj = posterior.sample((100_000,), x=x0)

    X_abc = []
    Y_abc = []
    
    if bounds is not None:
        adj = torch.clamp(adj, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values
    
    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt([2])

    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)

    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break
        if args.task == "bernoulli_glm":
            Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        else:
            Y_chunk = param_box(UnifSample(bins = 10), adj, num=nums)
        
        X_chunk = simulators(Y_chunk)
        
        index_ABC = ABC_rej2(x0, X_chunk, tol0, device, args.task)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    print("X_abc size", X_abc.size())

    task_benchmark = ["two_moons"]
    if args.task in task_benchmark:
        post_sample = true_posteriors(j = args.x0_ind+1)
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)
    
    tol = (args.tol/tol0 + 1e-12)

    density_estimator_npe = saved_data["density_estimator"]
    N = 10000
    
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()

    # 3) Put your data on the same device
    X_abc = X_abc.to(device)       # [B, x_dim]
    
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    samples_all = forward_from_Z_chunked(
        density_estimator_npe_gpu, 
        X_abc,
        Y_abc.size(1),
        N
    )
    samples_all = samples_all.cpu()
    mean_X = torch.mean(samples_all,0)
    covs_X = covs_chunked(samples_all, chunk=1000)
    covs_X = covs_X
    print(covs_X)

    samples_GPU = forward_from_Z_chunked(density_estimator_npe_gpu, x0, Y_abc.size(1), 50000)
    samples_GPU = samples_GPU.squeeze(1).cpu()
    mean_obs = torch.mean(samples_GPU,0)
    cov_obs = torch.cov(samples_GPU.T)
    print(mean_obs)
    print(cov_obs)
    sd_x0 = torch.linalg.cholesky(cov_obs)

    sd_X = torch.linalg.cholesky(covs_X)      # (10000, 2, 2)
    identity = torch.eye(sd_X.size(-1), device=sd_X.device)
    Sigma_inv_half = torch.linalg.solve_triangular(sd_X, identity, upper=False) # Getting inverse
    resid_tmp = Y_abc - mean_X
    resid_unsqueezed = resid_tmp.unsqueeze(-1)  # (B, d) → (B, d, 1)
    output = torch.bmm(Sigma_inv_half, resid_unsqueezed)  # (B, d, 1)
    output = output.squeeze(-1)
    new_theta = torch.matmul(output, sd_x0.squeeze(0).T) + mean_obs
    end_time = time.time()
    
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("NABC sample size: ", new_theta.size())
    results_size = min(10_000, new_theta.size(0))

    tmp = c2st(post_sample[:results_size].cpu(), new_theta[:results_size] )
    print(tmp)    
    
    NABC_results.append(tmp)
    
    sci_str = format(tol*tol0, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"../depot_hyun/hyun/NPE_ABC/NPE_ABC_c2st_results/{args.task}/{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save to output_dir
    pairplot(post_sample, figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()

    pairplot(new_theta[:10000], figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    torch.save(NABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--L", type = int, default = 10_000_000,
                        help = "Calibration data size (default: 10M)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training", type=int, default=100_000, 
                        help="Number of training data of NPE (default: 100_000)")
    parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"task: {args.task}")
    print(f"num_training: {args.num_training}")
    print(f"tol: {args.tol}")
    print(f"cond_den: {args.cond_den}")