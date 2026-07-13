import torch
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import channel_binary, channel_ternary, channel_quaternary
from simulator import channel_2x2, channel_3x3, channel_4x4, channel_5x5, channel_6x6

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

@torch.no_grad()
def gibbs_rr_3x3(
    y_obs: torch.Tensor,              # shape (9,), order [00,01,02,10,11,12,20,21,22]
    p1: float = 0.5,
    p2: float = 0.5,
    alpha: torch.Tensor = None,       # Dirichlet prior shape (9,)
    n_iter: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed)

    y = y_obs.to(torch.int64).view(-1)
    assert y.numel() == 9, "y_obs must have 9 cells."
    if alpha is None:
        alpha = torch.ones(9, dtype=torch.float32)
    else:
        alpha = alpha.to(torch.float32).view(-1)
        assert alpha.numel() == 9

    R = channel_3x3(p1, p2).to(torch.float32)   # (9,9)
    n = int(y.sum().item())

    p = torch.full((9,), 1/9, dtype=torch.float32)

    kept = []
    z = torch.zeros(9, 9, dtype=torch.int64)
    m = torch.distributions.Multinomial

    for t in range(n_iter):
        # Step 1: latent allocation
        W = R * p.view(1, 9)
        row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        W = W / row_sums

        for i in range(9):
            yi = int(y[i].item())
            if yi > 0:
                z[i] = m(total_count=yi, probs=W[i]).sample().to(torch.int64)
            else:
                z[i] = torch.zeros(9, dtype=torch.int64)

        # Step 2: update p
        z_true = z.sum(dim=0).to(torch.float32)  # (9,)
        p = torch.distributions.Dirichlet(alpha + z_true).sample()

        if t >= burn and ((t - burn) % thin == 0):
            kept.append(p.clone())

        if (t + 1) % 1000 == 0:
            print(f"Iteration {t+1}/{n_iter}", flush=True)

    p_samples = torch.stack(kept, dim=0)  # (S, 9)

    info = {
        "R": R,
        "n": n,
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "kept": p_samples.size(0),
    }
    return p_samples, info


@torch.no_grad()
def gibbs_rr_4x4(
    y_obs: torch.Tensor,              # shape (16,), order [00,01,02,03,10,...,33]
    p1: float = 0.5,
    p2: float = 0.5,
    alpha: torch.Tensor = None,       # Dirichlet prior shape (16,)
    n_iter: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed)
    y = y_obs.to(torch.int64).view(-1)
    assert y.numel() == 16, "y_obs must have 16 cells."
    if alpha is None:
        alpha = torch.ones(16, dtype=torch.float32)
    else:
        alpha = alpha.to(torch.float32).view(-1)
        assert alpha.numel() == 16
    R = channel_4x4(p1, p2).to(torch.float32)   # (16,16)
    n = int(y.sum().item())
    p = torch.full((16,), 1/16, dtype=torch.float32)
    kept = []
    z = torch.zeros(16, 16, dtype=torch.int64)
    m = torch.distributions.Multinomial
    for t in range(n_iter):
        # Step 1: latent allocation
        W = R * p.view(1, 16)
        row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        W = W / row_sums
        for i in range(16):
            yi = int(y[i].item())
            if yi > 0:
                z[i] = m(total_count=yi, probs=W[i]).sample().to(torch.int64)
            else:
                z[i] = torch.zeros(16, dtype=torch.int64)
        # Step 2: update p
        z_true = z.sum(dim=0).to(torch.float32)  # (16,)
        p = torch.distributions.Dirichlet(alpha + z_true).sample()
        if t >= burn and ((t - burn) % thin == 0):
            kept.append(p.clone())
        if (t + 1) % 1000 == 0:
            print(f"Iteration {t+1}/{n_iter}", flush=True)
    p_samples = torch.stack(kept, dim=0)  # (S, 16)
    info = {
        "R": R,
        "n": n,
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "kept": p_samples.size(0),
    }
    return p_samples, info

@torch.no_grad()
def gibbs_rr_5x5(
    y_obs: torch.Tensor,              # shape (25,)
    p1: float = 0.5,
    p2: float = 0.5,
    alpha: torch.Tensor = None,       # Dirichlet prior shape (25,)
    n_iter: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed)
    y = y_obs.to(torch.int64).view(-1)
    assert y.numel() == 25, "y_obs must have 25 cells."
    if alpha is None:
        alpha = torch.ones(25, dtype=torch.float32)
    else:
        alpha = alpha.to(torch.float32).view(-1)
        assert alpha.numel() == 25

    R = channel_5x5(p1, p2).to(torch.float32)
    n = int(y.sum().item())
    p = torch.full((25,), 1/25, dtype=torch.float32)

    kept = []
    z = torch.zeros(25, 25, dtype=torch.int64)
    m = torch.distributions.Multinomial

    for t in range(n_iter):
        # Step 1: latent allocation
        W = R * p.view(1, 25)
        row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        W = W / row_sums

        for i in range(25):
            yi = int(y[i].item())
            if yi > 0:
                z[i] = m(total_count=yi, probs=W[i]).sample().to(torch.int64)
            else:
                z[i] = torch.zeros(25, dtype=torch.int64)

        # Step 2: update p
        z_true = z.sum(dim=0).to(torch.float32)
        p = torch.distributions.Dirichlet(alpha + z_true).sample()

        if t >= burn and ((t - burn) % thin == 0):
            kept.append(p.clone())

        if (t + 1) % 1000 == 0:
            print(f"Iteration {t+1}/{n_iter}", flush=True)

    p_samples = torch.stack(kept, dim=0)  # (S, 25)

    info = {
        "R": R,
        "n": n,
        "n_iter": n_iter,
        "burn": burn,
        "thin": thin,
        "kept": p_samples.size(0),
    }
    return p_samples, info


@torch.no_grad()
def gibbs_rr_6x6(
    y_obs: torch.Tensor,              # shape (36,)
    p1: float = 0.5,
    p2: float = 0.5,
    alpha: torch.Tensor = None,       # Dirichlet prior shape (36,)
    n_iter: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int = 0,
):
    torch.manual_seed(seed)
    y = y_obs.to(torch.int64).view(-1)
    assert y.numel() == 36, "y_obs must have 36 cells."
    if alpha is None:
        alpha = torch.ones(36, dtype=torch.float32)
    else:
        alpha = alpha.to(torch.float32).view(-1)
        assert alpha.numel() == 36

    R = channel_6x6(p1, p2).to(torch.float32)
    n = int(y.sum().item())
    p = torch.full((36,), 1/36, dtype=torch.float32)

    kept = []
    z = torch.zeros(36, 36, dtype=torch.int64)
    m = torch.distributions.Multinomial

    for t in range(n_iter):
        # Step 1: latent allocation
        W = R * p.view(1, 36)
        row_sums = W.sum(dim=1, keepdim=True).clamp_min(1e-12)
        W = W / row_sums

        for i in range(36):
            yi = int(y[i].item())
            if yi > 0:
                z[i] = m(total_count=yi, probs=W[i]).sample().to(torch.int64)
            else:
                z[i] = torch.zeros(36, dtype=torch.int64)

        # Step 2: update p
        z_true = z.sum(dim=0).to(torch.float32)
        p = torch.distributions.Dirichlet(alpha + z_true).sample()

        if t >= burn and ((t - burn) % thin == 0):
            kept.append(p.clone())

        if (t + 1) % 1000 == 0:
            print(f"Iteration {t+1}/{n_iter}", flush=True)

    p_samples = torch.stack(kept, dim=0)  # (S, 36)

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
    elif args.table_num ==4:
        task ="table_dp_44"
    elif args.table_num ==5:
        task ="table_dp_55"
    else:
        raise ValueError("table_num must be 2, 3, 4 or 5.")


    y = torch.load(f"/home/hyun18/depot_hyun/hyun/NPE_ABC/seeds/{task}_obs.pt")[i-1]
    y = torch.tensor(y, dtype = torch.int64)

    # Run Gibbs
    n_iter = 2000000
    print("Gibbs sampling for task:", task, "x0 index:", i)
    if args.table_num ==2:
        samples, _ = gibbs_rr_2x2(y, p1=0.8, p2=0.8, n_iter=n_iter, burn=int(n_iter/10), thin=50, seed=1)
    elif args.table_num ==3:
        samples, _ = gibbs_rr_3x3(y, p1=0.8, p2=0.8, n_iter=n_iter, burn=int(n_iter/10), thin=50, seed=1)
    elif args.table_num ==4:
        samples, _ = gibbs_rr_4x4(y, p1=0.8, p2=0.8, n_iter=n_iter, burn=int(n_iter/10), thin=50, seed=1)
    elif args.table_num ==5:
        samples, _ = gibbs_rr_5x5(y, p1=0.8, p2=0.8, n_iter=n_iter, burn=int(n_iter/10), thin=50, seed=1)
    else:
        raise ValueError("table_num must be 2, 3, 4 or 5.")
    samples = samples.to(torch.float32)

    # Randomly permute and take 10000
    idx = torch.randperm(samples.size(0))[:10000]
    samples = samples[idx]

    print(f"Number of samples: {samples.size(0)}")
    torch.save(samples, f"/home/hyun18/depot_hyun/hyun/NPE_ABC/seeds/{task}_post_{i}.pt")
    
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run my_slcp task")
    parser.add_argument('--i', type=int, required=True, help="Index of the x0 to use")
    parser.add_argument('--table_num', type=int, default = 2, help="Number of tables (default: 2)")
    args = parser.parse_args()

    # Call the function with the specified index
    main(args)
