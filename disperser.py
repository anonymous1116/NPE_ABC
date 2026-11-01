# disperser.py
"""Fixed, invertible dispersion of a d-dim summary s into D dims.

mode="pad":   u = (z, eta),  eta ~ N(0, sigma^2 I_{D-d});   s' = Q2 phi(Q1 u)
mode="embed": u = A z,       A: D x d, orthonormal columns; s' = Q2 phi(u)
with z = (s - mu)/sd and phi(h) = asinh(alpha h)/alpha (identity when alpha = 0).
In both modes s can be recovered exactly from s', so p(theta | s') = p(theta | s).
"""
import torch, argparse, os


def haar_orthogonal(D, k=None, g=None):
    """D x k matrix with Haar-random orthonormal columns (k = D gives a full orthogonal matrix)."""
    k = k or D
    Q, R = torch.linalg.qr(torch.randn(D, k, generator=g, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(R))).float()


def build_disperser(s_pilot, d_out, mode="pad", alpha=1.0, sigma=1.0, seed=0, path=None):
    """Make the fixed transformation from pilot summaries s_pilot (n, d). Run once per task, then always load it."""
    assert mode in ("pad", "embed")
    s_pilot = s_pilot.float()
    d = s_pilot.shape[1]
    assert d_out >= d
    g = torch.Generator().manual_seed(seed)
    params = {
        "mode": mode, "d": d, "D": d_out, "alpha": float(alpha),
        "sigma": float(sigma) if mode == "pad" else 0.0, "seed": seed,
        "mu": s_pilot.mean(0),
        "sd": s_pilot.std(0).clamp_min(1e-8),        # avoid dividing by zero for constant coordinates
        "M1": haar_orthogonal(d_out, None if mode == "pad" else d, g),   # Q1 (D x D) or A (D x d)
        "Q2": haar_orthogonal(d_out, g=g),
    }
    if path is not None:
        torch.save(params, path)
    return params


class Disperser:
    def __init__(self, params, device="cpu"):
        if isinstance(params, str):
            params = torch.load(params, weights_only=True)
        self.p = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in params.items()}
        self.device = torch.device(device)

    def _phi(self, h):
        a = self.p["alpha"]
        return torch.asinh(a * h) / a if a > 0 else h

    def _phi_inv(self, h):
        a = self.p["alpha"]
        return torch.sinh(a * h) / a if a > 0 else h

    @torch.no_grad()
    def __call__(self, s, generator=None):
        """s: (n, d) -> s': (n, D). Pass a seeded CPU generator to fix the noise (e.g. for x_obs)."""
        p = self.p
        z = (s.to(self.device, torch.float32) - p["mu"]) / p["sd"]
        if p["mode"] == "pad":
            eta = torch.randn(z.shape[0], p["D"] - p["d"], generator=generator)  # drawn on CPU, so a seeded CPU generator works on any device
            u = torch.cat([z, eta.to(self.device) * p["sigma"]], 1)
            h = u @ p["M1"].T
        else:
            h = z @ p["M1"].T
        return self._phi(h) @ p["Q2"].T

    @torch.no_grad()
    def recover(self, s_prime):
        """s' -> s. For checking invertibility."""
        p = self.p
        h = self._phi_inv(s_prime.to(self.device, torch.float32) @ p["Q2"])
        z = (h @ p["M1"])[:, :p["d"]]                # pad: Q1^T then keep the first d; embed: A^T
        return z * p["sd"] + p["mu"]


def make_dispersed_simulator(simulator, disperser, device="cpu"):
    disp = disperser if isinstance(disperser, Disperser) else Disperser(disperser, device)
    return lambda theta: disp(simulator(theta))


def save_observation(x_obs, disperser, seed, path):
    """Disperse one observation with a fixed noise draw and save it."""
    disp = disperser if isinstance(disperser, Disperser) else Disperser(disperser)
    x_obs_D = disp(x_obs, generator=torch.Generator().manual_seed(seed)).cpu()
    torch.save(x_obs_D, path)
    return x_obs_D




from simulator import Simulators, Priors, Bounds, observation_lists

def main(args):
    # Example usage of the Simulators class
    #simulators = simulator.Simulators(task="slcp")
    torch.set_manual_seed(12345)
    simulator = Simulators(args.task)
    prior = Priors(args.task)
    s_pilot = simulator(prior.sample((100_000,)))
    path = f"utils/files/disperser_{args.task}_D20_{args.mode}.pt"


    tmp = build_disperser(s_pilot, d_out=20, mode=args.mode, alpha=1.0, sigma=1.0, seed=0, path=path)
    obs_list = []
    for j in range(10):
        x0 = observation_lists(args.task)[j]
        if x0.ndim == 1:
            x0 = torch.reshape(x0, (1, x0.size(0)))
        new_obs = disp(x0)
        obs_list.append(new_obs[0])
    obs_list = torch.stack(obs_list)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(obs_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_disperse{args.disperse_num}_obs.pt")
    print(f"Disperser saved to {path} and dispersed observations saved to {current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_disperse{args.disperse_num}_obs.pt")

    # check (once):
    disp = Disperser(path)
    s = simulator(prior.sample((1000,)))
    print("max recover error:", (disp.recover(disp(s)) - s).abs().max().item())


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument('--task', type=str, default='twomoons', 
                            help='Simulation type: Lapl, MoG')
    parser.add_argument("--mode", type=str, default='embed', 
                        help="Mode of disperser: embed, pad (default: embed)")
    parser.add_argument("--disperse_num", type=int, default=100, 
                        help="Disperser number (default: 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)  # Pass the entire args object to the main function
