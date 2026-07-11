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
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2
from torchdiffeq import odeint

# Use context for ABC compared with calibrating_flow.py I guess this is better

def main(args):
    seed = args.seed
    #torch.set_default_device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    priors = Priors(args.task)
    true_posteriors = true_Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)
    
    if args.task in ["slcp_distractors", "my_five_twomoons_err40"]:
        chunk_size = 10_000_000
    elif args.task in ["my_five_twomoons_err90"]:
        chunk_size = 5_000_000
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


    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device)
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]

    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/FMPE_nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator = saved_data["density_estimator"]
    posterior = saved_data["posterior"]
    
    density_estimator = density_estimator.to(device).eval()
    embed = density_estimator._embedding_net
    ode_fn = density_estimator.ode_fn

    # vector field MLP
    vf = density_estimator.net

    # Compute context from x_obs
    context = embed(x0.to(device))  # (1, context_dim) on GPU
    
    def reverse_ode(t, theta):
        # theta is already on GPU since theta_1 is on GPU
        t_tensor = (1 - t) * torch.ones(theta.shape[0], device=device)
        return -ode_fn(
            input=theta,
            condition=context.expand(theta.shape[0], -1),
            times=t_tensor
        )

    def forward_ode(t, theta):
        t_tensor = t * torch.ones(theta.shape[0], device=device)
        return ode_fn(
            input=theta,
            condition=context.expand(theta.shape[0], -1),
            times=t_tensor
        )
        
    ode_batch_size = 500
    z_tmp_list = []

    with torch.no_grad():
        for start in range(0, Y_cal.size(0), ode_batch_size):
            end = min(start + ode_batch_size, Y_cal.size(0))
            Y_batch = Y_cal[start:end].to(device)
            
            z_batch = odeint(
                reverse_ode,
                Y_batch,
                t=torch.linspace(0, 1, 100, device=device),
                method='dopri5',
                atol=1e-7,
                rtol=1e-7,
            )[-1]
            z_tmp_list.append(z_batch.cpu())
            del Y_batch, z_batch
            torch.cuda.empty_cache()

    z_tmp = torch.cat(z_tmp_list, dim=0)
    
    adj_list = []

    with torch.no_grad():
        for start in range(0, z_tmp.size(0), ode_batch_size):
            end = min(start + ode_batch_size, z_tmp.size(0))
            z_batch = z_tmp[start:end].to(device)
            
            adj_batch = odeint(
                forward_ode,
                z_batch,
                t=torch.linspace(0, 1, 100, device=device),
                method='dopri5',
                atol=1e-7,
                rtol=1e-7,
            )[-1]
            adj_list.append(adj_batch.cpu())
            del z_batch, adj_batch
            torch.cuda.empty_cache()

    adj = torch.cat(adj_list, dim=0)
    
    
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
        
        x0_embed = embed(x0.to(device))
        X_chunk_embed = embed(X_chunk.to(device))
    

        index_ABC = ABC_rej2(x0_embed, X_chunk_embed, args.tol*100, device)
        X_chunk, Y_chunk = X_chunk[index_ABC], Y_chunk[index_ABC]
        X_abc.append(X_chunk)
        Y_abc.append(Y_chunk)
        print(f"{i}th iteration out of {num_chunks}", flush = True)

        
    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)    

    X_abc_embed = embed(X_abc.to(device))
    index_ABC = ABC_rej2(x0_embed, X_abc_embed, 0.01, device)
    X_abc, Y_abc = X_abc[index_ABC], Y_abc[index_ABC]

    print("X_abc size", X_abc.size())

    if args.task in task_benchmark:
        post_sample = true_posteriors(j = args.x0_ind+1)
    elif args.task in ["my_five_twomoons"]:    
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)
    
    print("post_sample size", post_sample.size())
    if (1==0):

        with torch.no_grad():
            tmp, _ =  transform.forward(Y_abc.to(device), context = embed(X_abc.to(device)) )
            new_theta, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    

        new_theta = new_theta.cpu()
        # 4) Now call your fast function (or sbi’s sample_batched) on GPU
        end_time = time.time()
        
        
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print("TABC sample size: ", new_theta.size())
        results_size = min(10_000, new_theta.size(0))

        tmp = c2st(post_sample[:results_size].cpu(), new_theta[:results_size] )
        print(tmp)    
        
        NABC_results.append(tmp)
        
        sci_str = format(args.tol, ".0e")
        print(sci_str)  # Output: '1e-02'
        

        output_dir = f"../depot_hyun/hyun/NPE_ABC/flow_c2st_results/{args.task}_context/J_{int(args.num_training/1000)}K/{int(args.L/1_000_000)}M_eta{sci_str}"
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
        if torch.cuda.is_available():
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