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
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2, forward_from_Z_chunked, covs_chunked

def WABC_rejection(x0, X_cal, tol, theta_test, density_estimator, device):
    # 1) Compute distances on GPU
    x0 = x0.to(device)

    Z_test = forward_from_Z_chunked(density_estimator, X_cal, theta_test)
    
    mean_test = torch.mean(Z_test,dim =0)
    covs_test = covs_chunked(Z_test)

    L, _ = torch.linalg.eigh(covs_test)          # [B, p]
    L_sqrt = L.clamp(min=0).sqrt()            # [B, p]
    frob_sq = ((L_sqrt - 1) ** 2).sum(dim=-1) # [B]
        

    W_distances = torch.sqrt((mean_test ** 2 + frob_sq))
    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(W_distances, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (W_distances <= ds)
    torch.cuda.empty_cache()
    # Select points within tolerance and return to CPU if needed
    return wt1.cpu()

    

def main(args):
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    NABC_results = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    priors = Priors(args.task)
    true_posteriors = true_Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    if args.task in ["slcp_distractors"]:
        chunk_size = 10_000_000
    else:
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
    
    Z_init = torch.randn((10000,10))
    with torch.no_grad():
        theta_test, _ = transform.inverse(Z_init.to(device), context = embed(x0.expand((Z_init.size(0),x0.size(1))).to(device)))

    index_ABC = WABC_rejection(x0, X_cal, 1e-2, theta_test, density_estimator_npe, device)
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]

    print(X_cal.size())

    #flow = density_estimator_npe_gpu.net
    #transform=flow._transform
    #embed = flow._embedding_net
    #with torch.no_grad():
    #    tmp, _ =  transform.forward(Y_cal.to(device), context = embed(X_cal.to(device)) )
    #    adj, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
    #adj = adj.cpu()

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