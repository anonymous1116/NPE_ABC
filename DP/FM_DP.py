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
import inspect
from calibrating_DP import rtrunc_beta1b_torch, truncated_dirichletK_stick, truncated_dirichlet_batch_K

# Use context for ABC compared with calibrating_flow.py I guess this is better
def make_ode_functions(ode_fn, device):
    def forward_ode(t, theta, condition):
        t_tensor = t * torch.ones(theta.shape[0], device=device)
        return ode_fn(input=theta, condition=condition, times=t_tensor)

    def reverse_ode(t, theta, condition):
        t_tensor = t * torch.ones(theta.shape[0], device=device)
        return ode_fn(input=theta, condition=condition, times=t_tensor)

    return forward_ode, reverse_ode

def batched_odeint(ode_func, y0, condition, t_span, device, 
                   batch_size=500, method='dopri5', atol=1e-7, rtol=1e-7):
    """
    Run odeint in batches with per-sample conditioning.

    Args:
        ode_func:   function (t, theta, condition) -> d_theta/dt
        y0:         (N, d) initial conditions on CPU
        condition:  (N, ctx_dim) per-sample context on CPU
        t_span:     time points tensor on device
        device:     torch device
        batch_size: number of samples per batch

    Returns:
        (N, d) tensor on CPU
    """
    results = []

    with torch.no_grad():
        for start in range(0, y0.size(0), batch_size):
            end = min(start + batch_size, y0.size(0))
            y_batch   = y0[start:end].to(device)
            cond_batch = condition[start:end].to(device)

            # Close over cond_batch for this batch
            def ode_func_batch(t, theta, _cond=cond_batch):
                return ode_func(t, theta, _cond)

            out = odeint(
                ode_func_batch,
                y_batch,
                t=t_span,
                method=method,
                atol=atol,
                rtol=rtol,
            )[-1]
            results.append(out.cpu())

            del y_batch, cond_batch, out
            torch.cuda.empty_cache()

    return torch.cat(results, dim=0)

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
    
    if args.task == "table_dp_22":
        data_dim = 3
    elif args.task == "table_dp_33":
        data_dim = 8
    elif args.task == "table_dp_44":
        data_dim = 15
    elif args.task == "table_dp_55":
        data_dim = 24
    elif args.task == "table_dp_66":
        data_dim = 35
    

    chunk_size = 10_000_000
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

    X_cal = torch.clone(X_cal[:,:data_dim])
    Y_cal = torch.clone(Y_cal[:,:data_dim])


    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device)
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]

    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/FMPE_nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator = saved_data["density_estimator"]
    posterior = saved_data["posterior"]
    
    
    density_estimator = density_estimator.to(device).eval()
    ode_fn = density_estimator.ode_fn

    forward_ode, reverse_ode = make_ode_functions(ode_fn, device)

    # Forward: t: 1->0
    t_span_forward = torch.linspace(1, 0, 100, device=device)
    # Reverse: t: 0->1
    t_span_reverse = torch.linspace(0, 1, 100, device=device)

    # x0 condition — raw, not embedded
    x0_condition = x0.to(device)  # (1, d_x)

    # ----------------------------------------------------------------
    # Step 1: adj = T(T^{-1}(Y_cal; X_cal); x0) for prior box
    # ----------------------------------------------------------------

    # Reverse: Y_cal -> z, conditioned on raw X_cal
    z_tmp = batched_odeint(reverse_ode, Y_cal, X_cal, t_span_reverse, device, 
                           batch_size=100, atol=1e-6, rtol=1e-5)

    # Forward: z -> adj, conditioned on raw x0
    x0_expanded = x0_condition.cpu().expand(z_tmp.size(0), -1)  # (N, d_x)
    adj = batched_odeint(forward_ode, z_tmp, x0_expanded, t_span_forward, device,
                         batch_size=100, atol=1e-6, rtol=1e-5)

    if bounds is not None:
        adj = torch.clamp(adj, min=torch.tensor(bounds)[:,0], max=torch.tensor(bounds)[:,1])

    with torch.no_grad():
        max_vals = torch.max(adj, 0).values
        min_vals = torch.min(adj, 0).values

    slack = 0.005
    max_vals = (max_vals + slack).clamp(max=1.0)
    min_vals = (min_vals - slack).clamp(min=0.0)

    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)


    print("max_vals:", max_vals)
    print("min_vals:", min_vals)

    # ----------------------------------------------------------------
    # Step 2: Generate calibration data Y_abc, X_abc
    # ----------------------------------------------------------------
    X_abc_list = []
    Y_abc_list = []
    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt(2)

    for i in range(num_chunks + 1):
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end - start
        if nums == 0:
            break

        Y_chunk = truncated_dirichlet_batch_K(L =nums, K = data_dim+1 ,lower = min_vals, upper =max_vals).to(dtype=torch.float32)

        
        X_chunk = simulators(Y_chunk)

        index_ABC = ABC_rej2(x0.to(device), X_chunk.to(device), args.tol * 100, device)
        X_abc_list.append(X_chunk[index_ABC])
        Y_abc_list.append(Y_chunk[index_ABC])
        print(f"{i}th iteration out of {num_chunks}", flush=True)

    X_abc = torch.cat(X_abc_list)
    Y_abc = torch.cat(Y_abc_list)

    index_ABC = ABC_rej2(x0.to(device), X_abc.to(device), 0.01, device)
    X_abc = X_abc[index_ABC]
    Y_abc = Y_abc[index_ABC]

    print("X_abc size", X_abc.size())

    # ----------------------------------------------------------------
    # Step 3: new_theta = T(T^{-1}(Y_abc; X_abc); x0)
    # ----------------------------------------------------------------
    z_tmp = batched_odeint(reverse_ode, Y_abc, X_abc, t_span_reverse, device,
                           batch_size=500, atol=1e-6, rtol=1e-5)

    x0_expanded_abc = x0_condition.cpu().expand(z_tmp.size(0), -1)
    new_theta = batched_odeint(forward_ode, z_tmp, x0_expanded_abc, t_span_forward, device,
                               batch_size=500, atol=1e-6, rtol=1e-5)
    new_theta = new_theta.cpu()

    end_time = time.time()
    elapsed_time = end_time - start_time

    post_sample = true_posteriors(j=args.x0_ind + 1)
    
    print("TABC sample size: ", new_theta.size())
    results_size = min(10_000, new_theta.size(0))
    tmp = c2st(post_sample[:results_size].cpu(), new_theta[:results_size])
    print(tmp)

    TABC_results = []    
    TABC_results.append(tmp)

    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'

    output_dir = f"../depot_hyun/hyun/NPE_ABC/FM_flow_c2st_results/{args.task}_context/J_{int(args.num_training/1000)}K/{int(args.L/1_000_000)}M_eta{sci_str}"
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
    
    torch.save(TABC_results, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")
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
    parser.add_argument("--tol", type=float, default=1e-3,
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