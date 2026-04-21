import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors, task_benchmark
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot

def main(args):
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    NABC_results = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    priors = Priors(args.task)
    true_posteriors = true_Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
     
    start_time = time.time()
    x0 = observation_lists(args.task)[args.x0_ind]

    print(x0)
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1,x0.size(0)))
        
    chunk_size_cal = 10_000
    print("x0_size", x0.size(), flush = True)
    
    Y_cal = priors.sample((chunk_size_cal,))
    X_cal = simulators(Y_cal)

    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator_npe = saved_data["density_estimator"]
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()
    flow = density_estimator_npe_gpu.net
    transform=flow._transform
    embed = flow._embedding_net
    with torch.no_grad():
        tmp, _ =  transform.forward(Y_cal.to(device), context = embed(X_cal.to(device)) )
        adj, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
    adj = adj.cpu()

    X_abc = torch.clone(X_cal); Y_abc = torch.clone(adj)

    if args.task in task_benchmark:
        post_sample = true_posteriors(j = args.x0_ind+1)
    elif args.task in ["my_five_twomoons"]:    
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)
    
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    end_time = time.time()
    
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("NABC sample size: ", Y_abc.size())
    results_size = min(10_000, Y_abc.size(0))

    tmp = c2st(post_sample[:results_size].cpu(), Y_abc[:results_size] )
    print(tmp)    
    
    NABC_results.append(tmp)
    
    output_dir = f"../depot_hyun/hyun/NPE_ABC/flow_c2st_results_exp2/{args.task}_context/J_{int(args.num_training/1000)}K"
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

    pairplot(Y_abc[:10000], figsize=(6,6), limits = bounds)
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
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training", type=int, default=100_000, 
                        help="Number of training data of NPE (default: 100_000)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"task: {args.task}")
    print(f"num_training: {args.num_training}")
    print(f"cond_den: {args.cond_den}")