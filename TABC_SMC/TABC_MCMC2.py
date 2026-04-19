import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
import arviz

from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2, compute_mad, TABC_Jacobian


def sample_until_close(epsilon, mad, posterior, simulators, x0, device, batch_size=100):
    """
    Sample (theta, s) pairs until we find one with dist < epsilon.
    Returns the first accepted (theta, s) pair.
    """
    n_generated = 0
    while True:
        theta_cand = posterior.sample((batch_size,), x=x0, show_progress_bars=False)
        s_cand = simulators(theta_cand)
        n_generated += batch_size

        dist = torch.sqrt(torch.mean(
            torch.abs(s_cand.to(device) - x0.to(device))**2 / mad**2, 1
        ))
        
        min_idx = torch.argmin(dist)
        if dist[min_idx] < epsilon:
            return theta_cand[min_idx], s_cand[min_idx], n_generated

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
    

    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator_npe = saved_data["density_estimator"]
    posterior = saved_data['posterior'].set_default_x(x0)
    
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()
    flow = density_estimator_npe_gpu.net
    transform=flow._transform
    embed = flow._embedding_net

    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    input_dir = f"../depot_hyun/hyun/NPE_ABC/MCMC/{args.task}/J_{int(args.num_training/1000)}K/eta{sci_str}/x0{args.x0_ind}_seed{args.seed}_result.pt"
    get_epsilon = torch.load(input_dir)    
    dist_max =get_epsilon["dist_max"]
    mad = get_epsilon["mad"]
    
    ESS_TARGET = 10_000
    CHECK_EVERY = 50   # do NOT check every iteration

    theta_init = posterior.sample((1,), x0, show_progress_bars=False)
    
    theta_list = []
    s_list = []
    theta_list.append(theta_init[0])
    s_list.append(x0[0])

    ess_history_min = []
    ess_history_median = []
    iter_history = []
    acc_history = []
    n_generated_total = 0

    accepted_count = 0
    total_iterations = 1000000 # 12000
    posterior = saved_data['posterior'].set_default_x(x0)

    

    for j in range(1, total_iterations):  # large upper bound
        theta_cand_0, s_cand_0, n_generated = sample_until_close(epsilon=dist_max, mad=mad, posterior=posterior, simulators=simulators,x0=x0,device=device,batch_size =100)
        n_generated_total += n_generated
        alpha = priors.log_prob(theta_cand_0) - priors.log_prob(theta_list[j-1]) \
            + posterior.log_prob(theta_list[j-1]) - posterior.log_prob(theta_cand_0) \
    #        + TABC_Jacobian(s_list[j-1], theta_list[j-1], x0, density_estimator_npe) \
    #        - TABC_Jacobian(s_cand_0, theta_cand_0, x0, density_estimator_npe) 
    
        
        alpha = torch.exp(alpha)
        alpha = torch.min(torch.tensor(1.0), alpha)

        accept = torch.bernoulli(alpha)

        if accept == 1:     
            accepted_count += 1
            #if s_cand_0.ndim == 1:
            #    s_cand_0 = s_cand_0.unsqueeze(0)
            #if theta_cand_0.ndim == 1:
            #    theta_cand_0 = theta_cand_0.unsqueeze(0)

            #with torch.no_grad():
            #    tmp, _ = transform.forward(theta_cand_0.to(device), context=embed(s_cand_0.to(device)))
            #    adj, _ = transform.inverse(tmp, context=embed(x0.to(device)))

            theta_list.append(theta_cand_0.cpu())
            s_list.append(s_cand_0.cpu())
        
        else:
            theta_list.append(theta_list[j-1])
            s_list.append(s_list[j-1])
    
        # ESS check
        if j % CHECK_EVERY == 0:
            theta_chain = torch.row_stack(theta_list).cpu().numpy()
            theta_chain = arviz.convert_to_dataset(theta_chain[None,:,:])
            ess = arviz.ess(theta_chain, method="bulk")
            
            ess_min = ess.x.min().item()
            ess_median = ess.x.median().item()

            acc_rate = accepted_count / j
            ABC_acc_size = n_generated_total / j # average number of generated samples per iteration
            ess_history_min.append(ess_min)
            ess_history_median.append(ess_median)
            
            iter_history.append(j)
            acc_history.append(acc_rate)

            print(f"Iter {j}, ESS_min={ess_min:.1f}, ESS_median={ess_median:.1f}, acc={acc_rate:.3f}, ABC_acc_size={ABC_acc_size:.3f}", flush=True)
            if ess_median >= ESS_TARGET:
                break
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Elapsed time: {elapsed_time:.2f} seconds")
        if elapsed_time > 4*3600 - 20* 60:  # 3:40 hours
            time_limit_exceeded = True
            print("Time limit exceeded. Stopping the algorithm.")
            break
        else:
            time_limit_exceeded = False
    theta_stack= torch.row_stack(theta_list)
    s_stack= torch.row_stack(s_list)
    elapsed_time = end_time - start_time
    

    ran =torch.randint(int(len(theta_list)/5), len(theta_list), (10000,))
    #ran =torch.randint(0, len(theta_list), (10000,))
    sample_post_10K_MCMC = theta_stack[ran]
    s_10K = s_stack[ran]

    # calibrate
    with torch.no_grad():
        tmp, _ =  transform.forward(sample_post_10K_MCMC.to(device), context = embed(s_10K.to(device)) )
        adj, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    
    
    adj = adj.cpu()
    sample_post_10K = torch.clone(adj)
    ran2 = torch.randint(0, 10000, (1000,))
    sample_post_1K = adj[ran2]


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
    
    NPE_post_sample = posterior.sample((10000,), x0, show_progress_bars=False).cpu()
    c2st_NPE = c2st(post_sample.cpu(), NPE_post_sample.cpu())
    c2st_NPE_1K = c2st(post_sample[:1000].cpu(), NPE_post_sample[:1000].cpu())
    print(f"c2st_NPE: {c2st_NPE}, c2st_NPE_1K: {c2st_NPE_1K}")

    tmp = c2st(post_sample.cpu(), sample_post_10K.cpu())
    tmp2 = c2st(post_sample[:1000].cpu(), sample_post_1K.cpu())
    c2st_MCMC = c2st(post_sample.cpu(), sample_post_10K_MCMC.cpu())
    c2st_MCMC_1K = c2st(post_sample[:1000].cpu(), sample_post_10K_MCMC[ran2].cpu())
    
    print(f"c2st_10K: {tmp}, c2st_1K: {tmp2}, c2st_MCMC: {c2st_MCMC}, c2st_MCMC_1K: {c2st_MCMC_1K}")
    
    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"../depot_hyun/hyun/NPE_ABC/MCMC2_c2st_results/{args.task}/J_{int(args.num_training/1000)}K/eta{sci_str}"
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

    pairplot(sample_post_10K, figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    torch.save([tmp,tmp2], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
    
    torch.save({
        "config": {
        "x0_ind": args.x0_ind,
        "seed": args.seed,
        },
        "ESS_TARGET": ESS_TARGET,
        "CHECK_EVERY": CHECK_EVERY,
        "ess_history_min": ess_history_min,
        "ess_history_median": ess_history_median,
        "acc_history": acc_history,
        "time_limit_exceeded": time_limit_exceeded,
        "total_iterations": total_iterations,
        "elapsed_time": elapsed_time
    }, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_history.pt")
        


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