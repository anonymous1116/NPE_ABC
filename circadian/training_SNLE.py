import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from functions import simulators_circadian, y_to_lambda_batch
from help_functions import ABC_rej2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def simulate(theta, sim_device, batch_size):
    """Simulate in chunks on `sim_device`; return float32 CPU tensors."""
    xs = []
    for start in range(0, theta.shape[0], batch_size):
        th = theta[start:start + batch_size].to(sim_device)
        xs.append(simulators_circadian(th, device=sim_device).float().cpu())
        print(f"  sim chunk {start}: theta {tuple(th.shape)} on {th.device}", flush=True)
    return torch.cat(xs)


def drop_invalid(theta, x):
    """Remove pairs with NaN/Inf simulator output (e.g. ODE solver failures)."""
    mask = torch.isfinite(x).all(dim=1)
    n_bad = int((~mask).sum())
    if n_bad > 0:
        print(f"  dropped {n_bad} invalid simulations", flush=True)
    return theta[mask], x[mask]


def abc_proposal(prior, x0, n_keep, tol, sim_device, batch_size):
    """
    ABC rejection used ONLY to obtain proposal draws of theta.
    The selected (theta, x) pairs are discarded: selection on x truncates
    p(x | theta), so training NLE on them would bias the likelihood.
    """
    total = n_keep * int(round(1 / tol))
    thetas = []
    for start in range(0, total, batch_size):
        n = min(batch_size, total - start)
        th = prior.sample((n,)).float().to(sim_device)
        x = simulators_circadian(th, device=sim_device)
        idx = ABC_rej2(x0.to(sim_device), x, tol, sim_device)
        thetas.append(th[idx].cpu())
        print(f"  ABC chunk {start // batch_size + 1}/{-(-total // batch_size)}", flush=True)
    return torch.cat(thetas).float(), total


def save_round(output_dir, args, r, inference, samples, timings, n_sims_total):
    tag = f"seed{args.seed}_x0{args.x0_ind}_round{r}"
    with open(os.path.join(output_dir, f"{tag}.pkl"), "wb") as f:
        pickle.dump({"inference": inference, "timings": timings,
                     "n_sims_total": n_sims_total}, f)
    torch.save(samples, os.path.join(output_dir, f"{tag}_samples.pt"))

    pairplot(samples, figsize=(8, 8))
    plt.savefig(os.path.join(output_dir, f"plot_x0{args.x0_ind}_seed{args.seed}_round{r}.png"))
    plt.close()
    print(f"Round {r} saved | timings (s): {timings} | total sims so far: {n_sims_total}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # slice_np_vectorized uses NumPy's RNG
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    sim_device = torch.device(args.sim_device)
    print(f"[setup] torch threads={torch.get_num_threads()}, sim_device={sim_device}", flush=True)
    
    # prior (CPU)
    prior = BoxUniform(
        low=torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1, 1e-1]),
        high=torch.tensor([0.5, 3.5, 0.6, 0.5, 0.9, 0.8, 1.0, 9.0, 20.0]),
    )

    # observation
    y_mat = np.loadtxt("./circadian/y_mat.txt")
    x0s = y_to_lambda_batch(torch.tensor(y_mat)).float()
    x0 = x0s[args.x0_ind].reshape(1, -1)

    # output directory
    output_dir = (f"circadian/nets_depot/SNLE/{args.task}/"
                  f"round0_{args.round0}/N_{args.num_training}")
    os.makedirs(output_dir, exist_ok=True)

    mcmc_kwargs = dict(
        mcmc_method="slice_np_vectorized",
        mcmc_parameters={"num_chains": args.num_chains, "thin": args.thin,
                         "warmup_steps": args.warmup_steps, "init_strategy": "resample"},
    )

    inference = NLE(prior, density_estimator="nsf", device="cpu")
    n_sims_total = 0

    # ---------------- round 0: proposal draws ----------------
    t0 = time.time()
    if args.round0 == "abc":
        theta, n_abc = abc_proposal(prior, x0, args.num_training, args.tol,
                                    sim_device, args.sim_batch_size)
        n_sims_total += n_abc
    else:  # "prior"
        theta = prior.sample((args.num_training,)).float()
    print(f"[round0] theta shape={tuple(theta.shape)}, min={theta.min(0).values}, max={theta.max(0).values}", flush=True)
    t_proposal = time.time() - t0

    # ---------------- rounds 0..R ----------------
    for r in range(args.num_rounds + 1):
        timings = {}
        if r == 0:
            timings["proposal"] = t_proposal

        # fresh simulations at proposal theta (valid NLE training data)
        t0 = time.time()
        x = simulate(theta, sim_device, args.sim_batch_size)
        timings["simulate"] = time.time() - t0
        n_sims_total += theta.shape[0]
        theta, x = drop_invalid(theta, x)

        # train (pools data across rounds; no proposal correction needed for NLE)
        print(f"[r{r}] training on theta {tuple(theta.shape)} {theta.dtype}, x {tuple(x.shape)} {x.dtype}", flush=True)
        t0 = time.time()
        inference.append_simulations(theta, x).train()
        timings["train"] = time.time() - t0

        # posterior + MCMC sampling
        t0 = time.time()
        posterior = inference.build_posterior(**mcmc_kwargs)
        print(f"[r{r}] sampling posterior...", flush=True)
        samples = posterior.sample((args.num_training,), x=x0)
        print(f"[r{r}] samples shape={tuple(samples.shape)}, mean={samples.mean(0)}", flush=True)
        timings["sample"] = time.time() - t0

        save_round(output_dir, args, r, inference, samples, timings, n_sims_total)

        # this round's posterior samples are the next round's proposal draws
        theta = samples.float()

    print("Training completed.", flush=True)


def get_args():
    p = argparse.ArgumentParser(description="SNLE on the circadian model.")
    p.add_argument("--task", type=str, default="circadian")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_training", type=int, default=10_000,
                   help="Simulations per round (and posterior samples per round)")
    p.add_argument("--x0_ind", type=int, default=0)
    p.add_argument("--num_rounds", type=int, default=4,
                   help="Sequential rounds after round 0")
    p.add_argument("--round0", type=str, default="abc", choices=["abc", "prior"],
                   help="abc: ABC-accepted theta as proposal (x re-simulated); prior: plain prior")
    p.add_argument("--tol", type=float, default=0.01, help="ABC acceptance fraction")
    p.add_argument("--sim_device", type=str, default="cpu", help="cpu or cuda:0 (simulator only)")
    p.add_argument("--sim_batch_size", type=int, default=100_000)
    p.add_argument("--num_chains", type=int, default=50)
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())