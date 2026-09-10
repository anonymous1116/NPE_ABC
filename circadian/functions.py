
import torch
import numpy as np
from torchdiffeq import odeint


def simulators_circadian(theta, device="cpu", max_ODEtime=500, T_field=66, _depth=0):
    """
    Simulates the circadian model for a batch of parameter sets theta.
    On underflow (stiff/extreme parameter draw), recursively bisects the
    batch and retries each half, so only the truly problematic rows are lost.
    """
    M_obs_time = np.arange(max_ODEtime - T_field, max_ODEtime)
    batch_size = theta.size(0)

    if batch_size == 0:
        return torch.empty((0, 7), device=device)  # adjust 7 to your n_freqs

    y0 = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)
    t_eval = torch.arange(1, max_ODEtime + 1, dtype=torch.float64, device=device)

    theta_full = torch.column_stack([
        torch.ones(batch_size, device=device) * 24.44,
        theta,
        torch.ones(batch_size, device=device) * 8.0,
        torch.ones(batch_size, device=device) * 4.0,
    ])

    try:
        sol = odeint(
            lambda t, y: ode_model(t, y, theta_full),
            y0, t_eval, method="dopri5", rtol=1e-6, atol=1e-6,
        )
    except AssertionError:
        if batch_size == 1:
            # A single row is the culprit — give up on it, return NaNs so caller can filter it out
            print(f"[depth={_depth}] Dropping 1 unsolvable sample.", flush=True)
            return torch.full((1, 7), float("nan"), device=device)  # adjust 7 to n_freqs

        # Bisect the batch and retry each half independently
        mid = batch_size // 2
        left = simulators_circadian(theta[:mid], device, max_ODEtime, T_field, _depth + 1)
        right = simulators_circadian(theta[mid:], device, max_ODEtime, T_field, _depth + 1)
        return torch.cat([left, right], dim=0)

    ModelRun = sol.permute(1, 0, 2)
    M_batch = ModelRun[:, M_obs_time, 0]
    return y_to_lambda_batch(M_batch, deg=15).cpu()

def ode_model(t, y, theta):
    M, P, Pp = y[:, 0], y[:, 1], y[:, 2]
    nu, k1, k2, k3, k4, k5, k6, k7, Ka, Kb, m, n = theta.unbind(dim=1)

    dM = nu / (1 + (Pp / Ka) ** m) - k1 * M
    dP = k2 * M - (k3 + k4) * P + k6 * Pp - (k7 * P * Pp ** n) / (Kb ** n + Pp ** n)
    dPp = k4 * P - (k5 + k6) * Pp + (k7 * P * Pp ** n) / (Kb ** n + Pp ** n)

    return torch.stack([dM, dP, dPp], dim=1)

def make_fourier_design(T_y, deg=15, period=None, device="cpu", dtype=torch.float32):
    """Builds the same design matrix as fda::create.fourier.basis + eval.basis,
    with the constant column already dropped (matches basisMat[,-1])."""
    if period is None:
        period = T_y
    if deg % 2 == 0:
        deg += 1  # fda forces nbasis to be odd (const + sin/cos pairs)
    n_freqs = (deg - 1) // 2

    t = torch.arange(T_y, dtype=dtype, device=device)  # gr_x = 0,...,T_y-1
    omega = 2 * torch.pi / period

    cols = []
    for k in range(1, n_freqs + 1):
        cols.append(torch.sin(k * omega * t))
        cols.append(torch.cos(k * omega * t))
    X = torch.sqrt(torch.tensor(2.0 / period, dtype=dtype, device=device)) * torch.stack(cols, dim=1)
    return X  # (T_y, 2*n_freqs)


def y_to_lambda_batch(Y, deg=15, X_pinv=None):
    """
    Y: (batch, T_y) tensor of M trajectories
    Returns: (batch, n_freqs) tensor of S_sq per frequency
    """
    batch, T_y = Y.shape
    if X_pinv is None:
        X = make_fourier_design(T_y, deg, device=Y.device, dtype=Y.dtype)
        X_pinv = torch.linalg.pinv(X)  # (K, T_y), computed once, reused across batches

    beta = Y @ X_pinv.T                     # (batch, K) — the OLS coefficients, vectorized
    n_freqs = beta.shape[1] // 2
    beta = beta.view(batch, n_freqs, 2)      # group consecutive (sin_k, cos_k) pairs
    S_sq = (beta ** 2).sum(dim=2)            # (batch, n_freqs)
    return S_sq

