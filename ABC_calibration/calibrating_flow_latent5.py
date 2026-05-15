import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors, task_benchmark
from help_functions import UnifSample, compute_mad, param_box, truncated_mvn_sample, ABC_rej2, forward_from_theta_test, eigen_chunked

# To test TABC_rejection instead of WABC_rejection, replace the line 138 with the following line:
#index_ABC = TABC_rejection(x0, X_abc, 0.01, density_estimator_npe, Y_abc.size(1), device, num_samples=100


def ABC_rejection(x0, X_cal, tol, device, sort = False):
    # Move all tensors to the target device at once
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    mad = compute_mad(X_cal)
    mad = torch.reshape(mad, (1, X_cal.size(1))).to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2/mad**2, 1))
    
    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    torch.cuda.empty_cache()
    # Select points within tolerance and return to CPU if needed
    if sort:
        sorted_indices = torch.argsort(dist)
        return wt1.cpu(), sorted_indices.cpu()
    else:
        return wt1.cpu()



def WABC_rejection(x0, X_cal, tol, density_estimator, theta_dim, device, num_samples=1000, sort = False):
    Z_init = torch.randn((num_samples,theta_dim))
    density_estimator_npe_gpu = density_estimator.to(device).eval()
    flow = density_estimator_npe_gpu.net
    transform=flow._transform
    embed = flow._embedding_net

    del flow, density_estimator_npe_gpu
    with torch.no_grad():
        theta_test, _ = transform.inverse(Z_init.to(device), context = embed(x0.expand((Z_init.size(0),x0.size(1))).to(device)))

    Z_test = forward_from_theta_test(density_estimator, X_cal, theta_test)
    
    mean_test = torch.sum(torch.mean(Z_test,dim =0) ** 2, 1)
    frob_sq = eigen_chunked(Z_test)

    W_distances = torch.sqrt((mean_test + frob_sq))
    
    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(W_distances, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (W_distances <= ds)
    # Select points within tolerance and return to CPU if needed
    del transform, embed, Z_test, mean_test, frob_sq
    torch.cuda.empty_cache()
    if sort:
        sorted_indices = torch.argsort(W_distances)
        return wt1.cpu(), sorted_indices.cpu()
    else:
        return wt1.cpu()


def main(args):
    seed = args.seed
    #torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    TABC_results = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    priors = Priors(args.task)
    true_posteriors = true_Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    chunk_size = 1_000_000
    num_chunks = L // chunk_size
    
    start_time = time.time()
    x0 = observation_lists(args.task)[args.x0_ind]

    print(x0)
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1,x0.size(0)))
        
    chunk_size_cal = 10_000
    print("x0_size", x0.size(), flush = True)
    #print("X_cal size", X_cal.size(), flush = True)
    
    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)

    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator_npe = saved_data["density_estimator"]
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()

    flow = density_estimator_npe_gpu.net
    transform=flow._transform
    embed = flow._embedding_net
    
    #index_ABC = WABC_rejection(x0, X_cal, 1e-2, density_estimator_npe, Y_cal.size(1), device, num_samples=50)
    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device)
    
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]

    with torch.no_grad():
        tmp, _ =  transform.forward(Y_cal.to(device), context = embed(X_cal.to(device)) )
        adj, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
    adj = adj.cpu()

    X_abc = []
    Y_abc = []
    
    if bounds is not None:
        adj = torch.clamp(adj, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values
    
    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt(2)

    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)

    print(X_cal.size())
    new_tol = 1e-1
    
    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break

        if args.task.startswith("bernoulli_glm2"):
            Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        else:
            Y_chunk = param_box(UnifSample(bins = 10), adj, num=nums)
        
        X_chunk = simulators(Y_chunk)
        index_ABC = ABC_rej2(x0, X_chunk, args.tol/new_tol, device)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    index_WABC, rank_idx_WABC = WABC_rejection(x0, X_abc, new_tol, density_estimator_npe, Y_abc.size(1), device, num_samples=10, sort=True)
    X_abc_WABC, Y_abc_WABC = X_abc[index_WABC], Y_abc[index_WABC]

    index_ABC, rank_idx_ABC = ABC_rejection(x0, X_abc, new_tol, device, sort=True)
    X_abc_ABC, Y_abc_ABC = X_abc[index_ABC], Y_abc[index_ABC]


    print("X_abc size", X_abc.size())

    if args.task in task_benchmark:
        post_sample = true_posteriors(j = args.x0_ind+1)
    elif args.task in ["my_five_twomoons"]:    
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)
    
    with torch.no_grad():
        tmp, _ =  transform.forward(Y_abc_WABC.to(device), context = embed(X_abc_WABC.to(device)) )
        new_theta_WABC, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))   
        
        tmp, _ =  transform.forward(Y_abc_ABC.to(device), context = embed(X_abc_ABC.to(device)) )
        new_theta_ABC, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))   
        
    if bounds is not None:
        new_theta_WABC = torch.clamp(new_theta_WABC, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])
        tol_bound = 10000/new_theta_WABC.size(0)*new_tol
        # indices of accepted samples
        accepted_idx = rank_idx_WABC[:int(tol_bound*X_abc_WABC.size(0))]
        X_abc_WABC, Y_abc_WABC = X_abc[accepted_idx], Y_abc[accepted_idx]
        with torch.no_grad():
            tmp, _ =  transform.forward(Y_abc_WABC.to(device), context = embed(X_abc_WABC.to(device)) )
            new_theta_WABC, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
        new_theta_WABC = torch.clamp(new_theta_WABC, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])
    
        new_theta_ABC = torch.clamp(new_theta_ABC, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])
        tol_bound = 10000/new_theta_ABC.size(0)*new_tol
        # indices of accepted samples
        accepted_idx = rank_idx_ABC[:int(tol_bound*X_abc_ABC.size(0))]
        X_abc_ABC, Y_abc_ABC = X_abc[accepted_idx], Y_abc[accepted_idx]
        with torch.no_grad():
            tmp, _ =  transform.forward(Y_abc_ABC.to(device), context = embed(X_abc_ABC.to(device)) )
            new_theta_ABC, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
        new_theta_ABC = torch.clamp(new_theta_ABC, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    new_theta_WABC = new_theta_WABC.cpu()
    new_theta_ABC = new_theta_ABC.cpu()
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    end_time = time.time()
    
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("WABC sample size: ", new_theta_WABC.size())
    print("ABC sample size: ", new_theta_ABC.size())
    results_size = min(10_000, new_theta_WABC.size(0))
    results_size2 = min(10_000, new_theta_ABC.size(0))

    c2st_WABC = c2st(post_sample[:results_size].cpu(), new_theta_WABC[:results_size] )
    c2st_ABC = c2st(post_sample[:results_size2].cpu(), new_theta_ABC[:results_size2] )
    
    print("c2st_WABC:", c2st_WABC, "c2st_ABC:", c2st_ABC)    
    
    TABC_results.append([c2st_WABC, c2st_ABC])
    
    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    
    output_dir = f"../depot_hyun/hyun/NPE_ABC/flow_c2st_latent5/{args.task}_context/J_{int(args.num_training/1000)}K/{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # Save to output_dir
    pairplot(post_sample, figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference_bounds.png")
    plt.close()

    pairplot(post_sample, figsize=(6,6))
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()
    
    pairplot(new_theta_WABC[:10000], figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated_bounds.png")
    plt.close()
    
    pairplot(new_theta_WABC[:10000], figsize=(6,6))
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    torch.save(TABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    if device.type == 'cuda':
        torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")
    else:
        torch.save(["cpu", elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")

    #
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


#python ABC_calibration/calibrating_flow_latent3.py --x0_ind 1 --seed 1 --task "my_five_twomoons_err40" --num_training 3000000 --L 10000000 --tol 1e-3
