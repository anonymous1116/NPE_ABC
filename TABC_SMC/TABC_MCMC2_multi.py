import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
import arviz
from multiprocessing import Pool
from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2, compute_mad, TABC_Jacobian


def sample_until_close(epsilon, mad, posterior, embed, simulators, x0, device, batch_size=100):
    n_generated = 0
    while True:
        theta_cand = posterior.sample((batch_size,), x=x0, show_progress_bars=False)
        s_cand = simulators(theta_cand)
        n_generated += batch_size
        x0_embed = embed(x0.to(device))
        s_cand_embed = embed(s_cand.to(device))

        dist = torch.sqrt(torch.mean(
            torch.abs(s_cand_embed.to(device) - x0_embed.to(device))**2 / mad**2, 1
        ))

        min_idx = torch.argmin(dist)
        if dist[min_idx] < epsilon:
            return theta_cand[min_idx], s_cand[min_idx], n_generated


def run_single_chain(chain_args):
    """
    Runs a single MH chain. Designed to be called by multiprocessing.Pool.
    All setup is done inside to avoid pickling issues with sbi posterior.
    """
    chain_id, args, dist_max, mad, x0 = chain_args

    # Each process sets its own seed
    torch.manual_seed(args.seed + chain_id)
    np.random.seed(args.seed + chain_id)
    torch.set_default_device("cpu")

    device = torch.device("cpu")  # multiprocessing on CPU only

    # Re-load everything inside the process to avoid pickle issues
    priors = Priors(args.task)
    simulators = Simulators(args.task)

    output_file_path = os.path.join(
        f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{args.seed}_{args.cond_den}.pkl'
    )
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)

    posterior = saved_data['posterior'].set_default_x(x0)
    density_estimator_npe = saved_data["density_estimator"].to(device).eval()
    flow = density_estimator_npe.net
    embed = flow._embedding_net

    # Per-chain ESS target: divide total target across chains
    NUM_CHAINS = 16
    ESS_TARGET_PER_CHAIN = 10_000 // NUM_CHAINS  # = 1000 per chain
    CHECK_EVERY = 1_000

    theta_init = posterior.sample((1,), x0, show_progress_bars=False)
    theta_list = [theta_init[0]]
    s_list = [x0[0]]

    accepted_count = 0
    n_generated_total = 0
    total_iterations = 1_000_000
    time_limit_exceeded = False
    ess_history_min, ess_history_median, iter_history, acc_history = [], [], [], []

    start_time = time.time()
    print(f"[Chain {chain_id}] started on PID {os.getpid()}", flush=True)
    for j in range(1, total_iterations):
        theta_cand_0, s_cand_0, n_generated = sample_until_close(
            epsilon=dist_max, mad=mad, posterior=posterior, embed=embed,
            simulators=simulators, x0=x0, device=device, batch_size=1000
        )
        n_generated_total += n_generated

        alpha = priors.log_prob(theta_cand_0) - priors.log_prob(theta_list[j-1]) \
              + posterior.log_prob(theta_list[j-1]) - posterior.log_prob(theta_cand_0)

        alpha = torch.exp(alpha)
        alpha = torch.min(torch.tensor(1.0), alpha)
        accept = torch.bernoulli(alpha)

        if accept == 1:
            accepted_count += 1
            theta_list.append(theta_cand_0.cpu())
            s_list.append(s_cand_0.cpu())
        else:
            theta_list.append(theta_list[j-1])
            s_list.append(s_list[j-1])

        if j % CHECK_EVERY == 0:
            theta_chain = torch.row_stack(theta_list).cpu().numpy()
            theta_chain = arviz.convert_to_dataset(theta_chain[None, :, :])
            ess = arviz.ess(theta_chain, method="bulk")

            ess_min = ess.x.min().item()
            ess_median = ess.x.median().item()
            acc_rate = accepted_count / j
            abc_acc_size = n_generated_total / j

            ess_history_min.append(ess_min)
            ess_history_median.append(ess_median)
            iter_history.append(j)
            acc_history.append(acc_rate)

            print(f"[Chain {chain_id}] Iter {j}, ESS_min={ess_min:.1f}, ESS_median={ess_median:.1f}, "
                  f"acc={acc_rate:.3f}, ABC_acc_size={abc_acc_size:.3f}", flush=True)

            if ess_median >= ESS_TARGET_PER_CHAIN:
                print(f"[Chain {chain_id}] Target ESS Attained.")
                break

        elapsed_time = time.time() - start_time
        if elapsed_time > 4 * 3600 - 20 * 60:
            time_limit_exceeded = True
            print(f"[Chain {chain_id}] Time limit exceeded.")
            break

    theta_list = torch.row_stack(theta_list)
    s_list = torch.row_stack(s_list)
    
    # Burn-in: discard first 20%
    burn_in = int(len(theta_list) * 0.2)
    theta_list = theta_list[burn_in:]
    s_store     = s_list[burn_in:]

    # Randomly select 1000 samples per chain (10 chains × 1000 = 10K total)
    ran = torch.randint(0, len(theta_list), (1000,))
    theta_selected = theta_list[ran]
    s_selected     = s_store[ran]


    return {
        "chain_id": chain_id,
        "theta_selected": theta_selected,
        "s_selected": s_selected,
        "ess_history_min": ess_history_min,
        "ess_history_median": ess_history_median,
        "acc_history": acc_history,
        "iter_history": iter_history,
        "time_limit_exceeded": time_limit_exceeded,
        "n_generated_total": n_generated_total,
        "final_ess": ess_median
    }


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    priors = Priors(args.task)
    true_posteriors = true_Posteriors(args.task)
    simulators = Simulators(args.task)
    bounds = Bounds(args.task)

    start_time = time.time()
    x0 = observation_lists(args.task)[args.x0_ind]
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1, x0.size(0)))
    print("x0_size", x0.size(), flush=True)

    output_file_path = os.path.join(
        f'../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl'
    )
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator_npe = saved_data["density_estimator"]
    posterior = saved_data['posterior'].set_default_x(x0)
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()
    flow = density_estimator_npe_gpu.net
    transform = flow._transform
    embed = flow._embedding_net

    sci_str = format(args.tol, ".0e")
    input_dir = f"../depot_hyun/hyun/NPE_ABC/MCMC/{args.task}/J_{int(args.num_training/1000)}K/eta{sci_str}/x0{args.x0_ind}_seed{args.seed}_result.pt"
    get_epsilon = torch.load(input_dir, weights_only=True)
    dist_max = get_epsilon["dist_max"]
    mad = get_epsilon["mad"]
    print(mad, "mad")

    # ── Launch 10 parallel chains ──────────────────────────────────────────
    NUM_CHAINS = 16
    chain_args = [(c, args, dist_max, mad, x0) for c in range(NUM_CHAINS)]

    with Pool(processes=NUM_CHAINS) as pool:
        results = pool.map(run_single_chain, chain_args)

    # ── Combine chains ─────────────────────────────────────────────────────
    # Stack all chains: shape (NUM_CHAINS, N, D) for arviz R-hat
    all_theta = [torch.row_stack(r["theta_list"]) for r in results]
    all_s     = [torch.row_stack(r["s_list"])     for r in results]

    
    # Sample 1000 from each chain → 10K total
    samples_per_chain = 1000
    ran_indices = [torch.randint(0, len(t), (samples_per_chain,)) for t in all_theta]
    sample_post_10K_MCMC = torch.cat([t[idx] for t, idx in zip(all_theta, ran_indices)])
    s_10K                = torch.cat([s[idx] for s, idx in zip(all_s,     ran_indices)])
    
    s_10K = s_10K[torch.randint(0,s_10K.size(0), (10000,))]
    # R-hat diagnostic across chains
    min_len = min(len(t) for t in all_theta)
    theta_chains_np = np.stack([t[:min_len].cpu().numpy() for t in all_theta])  # (C, N, D)
    rhat_data = arviz.convert_to_dataset(theta_chains_np)
    rhat = arviz.rhat(rhat_data)
    print("R-hat:", rhat)

    # ── Calibrate ──────────────────────────────────────────────────────────
    with torch.no_grad():
        tmp, _ = transform.forward(sample_post_10K_MCMC.to(device), context=embed(s_10K.to(device)))
        adj, _ = transform.inverse(tmp, context=embed(x0.expand((tmp.size(0), x0.size(1))).to(device)))

    adj = adj.cpu()
    sample_post_10K = torch.clone(adj)
    ran2 = torch.randint(0, 10000, (1000,))
    sample_post_1K = adj[ran2]

    # ── Reference posterior ────────────────────────────────────────────────
    task_benchmark = ["two_moons", "bernoulli_glm2", "slcp_summary_transform2", "double_slcp_summary_transform2"]
    if args.task in task_benchmark:
        post_sample = true_posteriors(j=args.x0_ind + 1)
    elif args.task in ["my_five_twomoons"]:
        post_sample = torch.load(f"../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_post_{args.x0_ind+1}.pt")
    else:
        post_sample = true_posteriors(torch.tensor(x0), n_samples=10_000, bounds=bounds)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # ── C2ST ───────────────────────────────────────────────────────────────
    tmp  = c2st(post_sample.cpu(), sample_post_10K.cpu())
    tmp2 = c2st(post_sample[:1000].cpu(), sample_post_1K.cpu())
    c2st_MCMC    = c2st(post_sample.cpu(), sample_post_10K_MCMC.cpu())
    c2st_MCMC_1K = c2st(post_sample[:1000].cpu(), sample_post_10K_MCMC[ran2].cpu())
    print(f"c2st_10K: {tmp}, c2st_1K: {tmp2}, c2st_MCMC: {c2st_MCMC}, c2st_MCMC_1K: {c2st_MCMC_1K}")

    # ── Save ───────────────────────────────────────────────────────────────
    output_dir = f"../depot_hyun/hyun/NPE_ABC/MCMC2_multis_results/{args.task}/J_{int(args.num_training/1000)}K/eta{sci_str}"
    os.makedirs(output_dir, exist_ok=True)

    pairplot(post_sample, figsize=(6, 6), limits=bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()

    pairplot(sample_post_10K, figsize=(6, 6), limits=bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()

    torch.save([tmp, tmp2, c2st_MCMC, c2st_MCMC_1K], f"{output_dir}/x0{args.x0_ind}_seed{args.seed}.pt")

    torch.save({
        "config": {"x0_ind": args.x0_ind, "seed": args.seed},
        "ESS_TARGET": 10_000,
        "CHECK_EVERY": 1_000,
        "ess_history_min": [r["ess_history_min"] for r in results],
        "ess_history_median": [r["ess_history_median"] for r in results],
        "acc_history": [r["acc_history"] for r in results],
        "time_limit_exceeded": any(r["time_limit_exceeded"] for r in results),
        "elapsed_time": elapsed_time,
        "rhat": rhat,
    }, f"{output_dir}/x0{args.x0_ind}_seed{args.seed}_history.pt")


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind",       type=int,   default=1)
    parser.add_argument("--seed",         type=int,   default=1)
    parser.add_argument('--task',         type=str,   default='twomoons')
    parser.add_argument("--num_training", type=int,   default=1_000_000)
    parser.add_argument("--tol",          type=float, default=1e-3)
    parser.add_argument('--cond_den',     type=str,   default='nsf')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)