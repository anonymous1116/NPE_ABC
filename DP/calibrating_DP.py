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
from help_functions import UnifSample, param_box, ABC_rej2

def rtrunc_beta1b_torch(size, beta, a=0.0, b=1.0, *, device=None, dtype=torch.float32, generator=None):
    if isinstance(size, int):
        size = (size,)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype).expand(size)
    a_t = torch.as_tensor(a, device=device, dtype=dtype).expand(size)
    b_t = torch.as_tensor(b, device=device, dtype=dtype).expand(size)

    if torch.any(beta_t <= 0):
        raise ValueError("beta must be > 0.")
    if torch.any((a_t < 0) | (a_t >= b_t) | (b_t > 1)):
        raise ValueError("Require 0 <= a < b <= 1.")

    A = torch.exp(beta_t * torch.log1p(-b_t))  # (1-b)^beta
    B = torch.exp(beta_t * torch.log1p(-a_t))  # (1-a)^beta

    U = torch.rand(size, device=device, dtype=dtype, generator=generator)
    U = A + (B - A) * U
    X = 1.0 - torch.pow(U, 1.0 / beta_t)
    return X



def truncated_dirichletK_stick(L, K, lower, upper, *, device=None, dtype=torch.float32, generator=None, eps=1e-12):
    """
    Draw L samples from Dirichlet(1,...,1) (K components) with component-wise truncation.
    Uses stick-breaking: theta_k = V_k * prod_{j<k}(1 - V_j), V_k ~ Beta(1, K-k).

    Args:
        L (int): number of samples
        K (int): dimension (e.g. 9 for 3x3, 16 for 4x4)
        lower, upper: length-K tensors with 0 <= lower[i] <= upper[i] <= 1
        device, dtype, generator: passed through to torch ops.

    Returns:
        (L, K) tensor
    """
    lower = torch.as_tensor(lower, device=device, dtype=dtype).reshape(-1)
    upper = torch.as_tensor(upper, device=device, dtype=dtype).reshape(-1)
    assert lower.numel() == K and upper.numel() == K, f"lower and upper must be length {K}."

    # Expand to (L, K)
    lower = lower.unsqueeze(0).expand(L, K).clone()
    upper = upper.unsqueeze(0).expand(L, K).clone()

    if torch.any(lower < 0) or torch.any(upper > 1) or torch.any(lower > upper):
        raise ValueError("Require 0 <= lower[i] <= upper[i] <= 1 for all i.")
    if torch.any(lower.sum(dim=1) - 1 > 1e-10):
        raise ValueError("Infeasible: sum(lower) must be <= 1.")
    if torch.any(1 - upper.sum(dim=1) > 1e-10):
        raise ValueError("Infeasible: sum(upper) must be >= 1.")

    thetas = []
    leftover = torch.ones(L, device=device, dtype=dtype)

    for k in range(K - 1):
        lk = lower[:, k]
        uk = upper[:, k]

        # Remaining lower/upper sums for components k+1,...,K-1
        rem_lower = lower[:, k+1:].sum(dim=1)
        rem_upper = upper[:, k+1:].sum(dim=1)

        # Bounds on V_k from theta_k in [lk, uk]
        ak_from_tk = (lk / leftover).clamp(0.0, 1.0)
        bk_from_tk = (uk / leftover).clamp(0.0, 1.0)

        # Bounds on V_k from feasibility of remaining components:
        # leftover * (1 - V_k) must be in [rem_lower, rem_upper]
        ak_from_rem = (1.0 - rem_upper / leftover.clamp_min(eps)).clamp(0.0, 1.0)
        bk_from_rem = (1.0 - rem_lower / leftover.clamp_min(eps)).clamp(0.0, 1.0)

        ak = torch.maximum(ak_from_tk, ak_from_rem)
        bk = torch.minimum(bk_from_tk, bk_from_rem)

        if torch.any(ak >= bk):
            raise ValueError(f"Infeasible bounds at step k={k}.")

        # V_k ~ Beta(1, K-k-1) truncated to [ak, bk]
        beta_param = float(K - k - 1)
        if beta_param > 0:
            Vk = rtrunc_beta1b_torch((L,), beta=beta_param, a=ak, b=bk,
                                     device=device, dtype=dtype, generator=generator)
        else:
            # Last stick: V_{K-1} ~ Uniform[ak, bk]
            Vk = ak + (bk - ak) * torch.rand(L, device=device, dtype=dtype, generator=generator)

        theta_k = leftover * Vk
        thetas.append(theta_k)
        leftover = (leftover * (1.0 - Vk)).clamp_min(eps)

    # Last component
    thetas.append(leftover)

    return torch.stack(thetas, dim=1)

def truncated_dirichlet_batch_K(L, K, lower, upper, *, batch_size=1_000_000, device=None, dtype=torch.float32):
    """
    Batched version of truncated_dirichletK_stick for large L.

    Args:
        L (int): total number of samples
        K (int): dimension (9 for 3x3, 16 for 4x4)
        lower, upper: length-K tensors
        batch_size: chunk size for memory efficiency

    Returns:
        (L, K) tensor on CPU
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output = []
    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        num = end - start
        if num == 0:
            break
        theta_batch = truncated_dirichletK_stick(
            num, K, lower, upper,
            device=device, dtype=dtype, generator=None, eps=1e-12
        )
        output.append(theta_batch.cpu())

    return torch.cat(output, dim=0)

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

    chunk_size = 10_000_000
    num_chunks = L // chunk_size
    
    start_time = time.time()
    x0 = observation_lists(args.task)[args.x0_ind]
    x0 = x0[:data_dim]  # Only take the first data_dim elements for calibration
    print(x0)
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1,x0.size(0)))
        
    print("x0_size", x0.size(), flush = True)
    
    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)


    X_cal = torch.clone(X_cal[:,:data_dim])
    Y_cal = torch.clone(Y_cal[:,:data_dim])

    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device)
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

    theta_minus1 = torch.ones(adj.size(0)) - torch.sum(adj, dim=1)
    theta_minus1 = torch.max(theta_minus1, torch.zeros_like(theta_minus1))  # Ensure non-negative

    adj = torch.column_stack((adj, theta_minus1))  # Append theta_{K-1} to adj
    
    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values
    
    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)

    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break

        Y_chunk = truncated_dirichlet_batch_K(L =nums, K = data_dim+1 ,lower = min_vals, upper =max_vals).to(dtype=torch.float32)
        X_chunk = simulators(Y_chunk)

        X_chunk = torch.clone(X_chunk[:,:data_dim])
        Y_chunk = torch.clone(Y_chunk[:,:data_dim])

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

    with torch.no_grad():
        tmp, _ =  transform.forward(Y_abc.to(device), context = embed(X_abc.to(device)) )
        new_theta, _ = transform.inverse(tmp, context = embed(x0.expand((tmp.size(0),x0.size(1))).to(device)))    

    new_theta = new_theta.cpu()
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    end_time = time.time()
    

    elapsed_time = end_time - start_time  # Calculate elapsed time
    post_sample = true_posteriors(j = args.x0_ind+1)
    
    print("TABC sample size: ", new_theta.size())
    results_size = min(10_000, new_theta.size(0))

    tmp = c2st(post_sample[:results_size].cpu(), new_theta[:results_size] )
    print(tmp)    
    TABC_results =  []
    TABC_results.append(tmp)
    
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