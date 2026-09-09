
import torch
from torchdiffeq import odeint


def simulators_circadian(theta, device = "cpu", max_ODEtime = 500, T_field = 66):
    """
    Simulates the circadian model for a batch of parameter sets theta.
    Args:
        theta: (batch, 12) tensor of parameters
    """

    # --- Settings ---
    M_obs_time = np.arange(max_ODEtime - T_field, max_ODEtime)  # (max_ODEtime-T_field+1):max_ODEtime

    # --- Initial conditions, replicated per batch item ---
    batch_size = theta.size(0)
    y0 = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)  # M, P, Pp all start at 0

    # --- Time points ---
    t_eval = torch.arange(1, max_ODEtime + 1, dtype=torch.float64, device=device)

    # --- Solve all trajectories at once ---
    # odeint's func signature is func(t, y) -> dy/dt; wrap theta_batch via closure

    theta = torch.column_stack([torch.ones(batch_size, device = device) * 24.44, 
                                theta, 
                                torch.ones(batch_size, device = device) * 8.0, 
                                torch.ones(batch_size, device = device) * 4.0])  
    
    sol = odeint(
        lambda t, y: ode_model(t, y, theta),
        y0,
        t_eval,
        method="dopri5",   # Dormand-Prince, same family as R's ode45
        rtol=1e-10,
        atol=1e-10,
    )
    # sol shape: (time, batch, 3)  ->  matches deSolve output per-batch-item if you index sol[:, i, :]

    ModelRun = sol.permute(1, 0, 2)  # (batch, time, 3) if you prefer batch-first
    M_batch = ModelRun[:, M_obs_time, 0]          # (batch, T_y), the M trajectories only
    return y_to_lambda_batch(M_batch, deg=15).cpu()  # (batch, n_freqs), the S_sq per frequency


def ode_model(t, y, theta):
    M, P, Pp = y[:, 0], y[:, 1], y[:, 2]
    nu, k1, k2, k3, k4, k5, k6, k7, Ka, Kb, m, n = theta.unbind(dim=1)

    dM = nu / (1 + (Pp / Ka) ** m) - k1 * M
    dP = k2 * M - (k3 + k4) * P + k6 * Pp - (k7 * P * Pp ** n) / (Kb ** n + Pp ** n)
    dPp = k4 * P - (k5 + k6) * Pp + (k7 * P * Pp ** n) / (Kb ** n + Pp ** n)

    return torch.stack([dM, dP, dPp], dim=1)

def make_fourier_design(T_y, deg=15, period=None, device="cpu", dtype=torch.float64):
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

