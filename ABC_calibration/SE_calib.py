import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
from sbi.inference import NPE, FMPE, NPSE
from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors, task_benchmark
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2
from torchdiffeq import odeint
import inspect

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

    if args.SDE == "ve":
        SDE_name = ""
    elif args.SDE == "vp":
        SDE_name = "vp_"
    else:
        raise ValueError("SDE name has to be determined")

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


    output_file_path = os.path.join(f'../depot_hyun/hyun/NPE_ABC/NPSE_{SDE_name}nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}.pkl')
    saved_data = torch.load(output_file_path.replace('.pkl', '.pt'), map_location='cpu')

    # Rebuild structure
    if args.SDE == "ve":
        inference = NPSE(prior=priors)
    elif args.SDE == "vp":
        inference = NPSE(prior=priors, sde_type="vp")
    else:
        raise ValueError("SDE name has to be determined")

    theta_tmp = priors.sample((10,))
    X_tmp = simulators(theta_tmp)
    inference.append_simulations(theta_tmp, X_tmp)
    density_estimator = inference.train(max_num_epochs=1)

    # Load ALL state including buffers
    density_estimator.load_state_dict(saved_data['full_state_dict'])
    density_estimator = density_estimator.eval()
    
    
    density_estimator = density_estimator.to(device).eval()
    ode_fn = density_estimator.ode_fn
    t_min = density_estimator.t_min
    t_max = density_estimator.t_max

    forward_ode, reverse_ode = make_ode_functions(ode_fn, device)

    # Forward: t: 1->0
    t_span_forward = torch.linspace(t_max, t_min, 100, device=device)
        
    # Reverse: t: 0->1
    t_span_reverse = torch.linspace(t_min, t_max, 100, device=device)

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

        if args.task.startswith("bernoulli_glm2"):
            Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        else:
            Y_chunk = param_box(UnifSample(bins=10), adj, num=nums)

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

    if args.task in task_benchmark:
        post_sample = true_posteriors(j=args.x0_ind + 1)
    elif args.task in ["my_five_twomoons"]:
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)

    print("TABC sample size: ", new_theta.size())
    results_size = min(10_000, new_theta.size(0))
    tmp = c2st(post_sample[:results_size].cpu(), new_theta[:results_size])
    print(tmp)

    TABC_results = []    
    TABC_results.append(tmp)

    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    

    output_dir = f"../depot_hyun/hyun/NPE_ABC/SE_{SDE_name}flow_c2st_results/{args.task}_context/J_{int(args.num_training/1000)}K/{int(args.L/1_000_000)}M_eta{sci_str}"
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
    parser.add_argument("--SDE", type=str, default= "ve",
                    help="ve or vp")
    
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