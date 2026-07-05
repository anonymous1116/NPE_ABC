import torch
import argparse
import os


# ---------- Channels ----------
def channel_binary(p: float) -> torch.Tensor:
    """
    One-bit randomized response:
      with prob p -> report truth; with prob 1-p -> random (0/1, each 1/2).
    Returns 2x2 matrix C with rows=reported, cols=true.
    """
    same = 0.5 * (1.0 + p)
    flip = 0.5 * (1.0 - p)
    C = torch.tensor([[same, flip],
                      [flip, same]], dtype=torch.float32)
    return C

def channel_2x2(p1: float, p2: float) -> torch.Tensor:
    """
    Joint channel R for two independent binary variables.
    Cell order is [00, 01, 10, 11] consistently.
    """
    C1 = channel_binary(p1)
    C2 = channel_binary(p2)
    R = torch.kron(C1, C2)  # (4,4), rows=reported, cols=true
    return R


# ---------- Gibbs sampler ----------
@torch.no_grad()
def gibbs_rr_2x2(
    y_obs: torch.Tensor,              # shape (4,), privatized counts in order [00,01,10,11]
    p1: float = 0.5,                  # truthful prob for var 1
    p2: float = 0.5,                  # truthful prob for var 2
    alpha: torch.Tensor = None,       # Dirichlet prior on true cell probs p (shape (4,))
    n_iter: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int = 0,
):
    """
    Gibbs sampling with latent allocation z_{i,k} for randomized response.

    Model:
      true probs p ~ Dir(alpha)
      reported counts y_obs ~ Multinomial(n, q), q = R @ p
      latent z_{i,k} | y_i, p ∼ Mult(y_i, w_{i,·}),  w_{i,k} ∝ p_k * R_{i,k}
      p | z ∼ Dir(alpha + z_{·,k})

    Returns:
      p_samples: (S, 4) samples of true cell probabilities p (S = kept draws)
      info: dict with R (4,4) and bookkeeping
    """
    torch.manual_seed(seed)

    # --- inputs & defaults ---
    y = y_obs.to(torch.int64).view(-1)          # (4,)
    assert y.numel() == 4, "y_obs must have 4 cells in order [00,01,10,11]."
    if alpha is None:
        alpha = torch.ones(4, dtype=torch.float32)  # Dirichlet(1,1,1,1)
    else:
        alpha = alpha.to(torch.float32).view(-1)
        assert alpha.numel() == 4

    R = channel_2x2(p1, p2).to(torch.float32)   # (4,4)
    n = int(y.sum().item())

    # --- init p (uniform is fine) ---
    p = torch.full((4,), 0.25, dtype=torch.float32)

    kept = []
    # Pre-allocate tensors to reduce overhead
    z = torch.zeros(4, 4, dtype=torch.int64)    # rows=reported i, cols=true k
    m = torch.distributions.Multinomial

    for t in range(n_iter):
        # (1) latent allocation: z_{i,·} ~ Mult(y_i, w_i),  w_i ∝ p_k * R_{i,k}
        # Compute unnormalized weights W (4x4): row i, col k
        W = (R * p.view(1, 4))                  # broadcast p across rows
        # Normalize each row to get probs
        row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        W = W / row_sums

        # Sample each row's allocation
        for i in range(4):
            yi = int(y[i].item())
            if yi > 0:
                zi = m(total_count=yi, probs=W[i]).sample().to(torch.int64)  # (4,)
            else:
                zi = torch.zeros(4, dtype=torch.int64)
            z[i] = zi

        # (2) update p | z  ~ Dir(alpha + z_{·,k})
        z_true = z.sum(dim=0).to(torch.float32)  # (4,)
        p = torch.distributions.Dirichlet(alpha + z_true).sample()

        # collect
        if t >= burn and ((t - burn) % thin == 0):
            kept.append(p.clone())

        if (t+1) % 1000 == 0:
            print(f"Iteration {t+1}/{n_iter}", flush= True)
        
    p_samples = torch.stack(kept, dim=0)  # (S,4)

    info = {
        "R": R,
        "n": n,
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "kept": p_samples.size(0),
    }
    return p_samples, info


def main(args):
    """
    Example: simulate privatized counts from known true p, then run Gibbs.
    """
    
    i = args.i
    
    if args.table_num ==2:
        task ="table_dp_22"
    elif args.table_num ==3:
        task ="table_dp_33"


    y = torch.load(f"/home/hyun18/depot_hyun/hyun/NPE_ABC/seeds/{task}/{task}_obs.pt")[i-1]
    y = torch.tensor(y, dtype = torch.int64)

    # Run Gibbs
    n_iter = 2000000
    print("Gibbs sampling for task:", task, "x0 index:", i)
    if args.table_num ==2:
        samples, _ = gibbs_rr_2x2(y, p1=0.8, p2=0.8, n_iter=n_iter, burn=int(n_iter/10), thin=50, seed=1)
    samples = samples.to(torch.float32)

    # Randomly permute and take 10000
    idx = torch.randperm(samples.size(0))[:10000]
    samples = samples[idx]

    print(f"Number of samples: {samples.size(0)}")
    torch.save(samples, f"/home/hyun18/depot_hyun/hyun/NPE_ABC/seeds/{task}/{task}_post_{i}.pt")
    
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run my_slcp task")
    parser.add_argument('--i', type=int, required=True, help="Index of the x0 to use")
    parser.add_argument('--table_num', type=int, default = 2, help="Number of tables (default: 2)")
    args = parser.parse_args()

    # Call the function with the specified index
    main(args)
