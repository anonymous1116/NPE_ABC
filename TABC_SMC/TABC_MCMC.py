import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
import arviz as az

from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2, compute_mad, TABC_Jacobian

def main(args):
    seed = args.seed
    torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
        
    print("x0_size", x0.size(), flush = True)
    

    # For inital value
    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)


    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device, args.task)
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]

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

    
    ESS_TARGET = 1_000
    CHECK_EVERY = 100   # do NOT check every iteration

    theta_list = []
    s_list = []
    theta_list.append(adj[0])
    s_list.append(X_cal[0])

    ess_history = []
    iter_history = []
    acc_history = []

    accepted_count = 0

    for j in range(1, 10000):  # large upper bound
        posterior = saved_data['posterior'].set_default_x(x0)
        theta_cand = posterior.sample((1000,), x=x0, show_progress_bars=False)

        s_cand = simulators(theta_cand)
        mad = compute_mad(s_cand)
        mad = torch.reshape(mad, (1, s_cand.size(1))).to(device)
        dist = torch.sqrt(torch.mean(torch.abs(s_cand.to(device) - x0.to(device))**2/mad**2, 1))        

        theta_cand_0 = theta_cand[torch.argmin(dist)]
        s_cand_0 = s_cand[torch.argmin(dist)]

        
        alpha = priors.log_prob(theta_cand_0) - priors.log_prob(theta_list[j-1]) \
            + posterior.log_prob(theta_list[j-1]) - posterior.log_prob(theta_cand_0) \
            + TABC_Jacobian(s_list[j-1], theta_list[j-1], x0, density_estimator_npe) \
            - TABC_Jacobian(s_cand_0, theta_cand_0, x0, density_estimator_npe) 
    
        
        alpha = torch.exp(alpha)
        alpha = torch.min(torch.tensor(1.0), alpha)

        accept = torch.bernoulli(alpha)

        if accept == 1:     
            accepted_count += 1
            if s_cand_0.ndim == 1:
                s_cand_0 = s_cand_0.unsqueeze(0)
            if theta_cand_0.ndim == 1:
                theta_cand_0 = theta_cand_0.unsqueeze(0)

            with torch.no_grad():
                tmp, _ = transform.forward(theta_cand_0.to(device), context=embed(s_cand_0.to(device)))
                adj, _ = transform.inverse(tmp, context=embed(x0.to(device)))

            theta_list.append(adj[0].cpu())
            s_list.append(s_cand_0.cpu())
        
        else:
            theta_list.append(theta_list[j-1])
            s_list.append(s_list[j-1])
    
        # ESS check
        if j % CHECK_EVERY == 0:
            theta_chain = torch.row_stack(theta_list).cpu().numpy()
            print(theta_chain.shape)
            ess = az.ess(theta_chain[None, :, :], method="bulk")
            ess_min = ess.min()

            acc_rate = accepted_count / j

            ess_history.append(ess_min)
            iter_history.append(j)
            acc_history.append(acc_rate)

            print(f"Iter {j}, ESS_min={ess_min:.1f}, acc={acc_rate:.3f}")
            if ess_min >= ESS_TARGET:
                break
    
        
    sample_post_10K = tmp[torch.randint(10000, len(theta_list), (10000,))]
    sample_post_1K = tmp[torch.randint(10000, len(theta_list), (1000,))]


    task_benchmark = ["two_moons", "bernoulli_glm2", "slcp_summary_transform2", "double_slcp_summary_transform2"]
    if args.task in task_benchmark:
        post_sample = true_posteriors(j = args.x0_ind+1)
    elif args.task in ["my_five_twomoons"]:    
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)
    
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    end_time = time.time()
    
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    tmp = c2st(post_sample.cpu(), sample_post_10K.cpu())
    tmp2 = c2st(post_sample[:1000].cpu(), sample_post_1K.cpu())
    print(f"c2st_10K: {tmp}, c2st_1K: {tmp2}")
    
    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"../depot_hyun/hyun/NPE_ABC/MCMC_c2st_results/{args.task}/J_{int(args.num_training/1000)}K/eta{sci_str}"
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

    pairplot(sample_post_10K[:10000], figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    torch.save([tmp,tmp2], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    torch.save([torch.cuda.get_device_name(0), elapsed_time], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_info.pt")

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training", type=int, default=1_000_000, 
                        help="Number of training data of NPE (default: 1_000_000)")
    parser.add_argument("--tol", type=float, default=1e-3,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
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
    print(f"tol: {args.tol}")
    print(f"cond_den: {args.cond_den}")