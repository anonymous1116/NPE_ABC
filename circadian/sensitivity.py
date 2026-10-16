import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import periodogram, get_window
from scipy.fft import next_fast_len
import argparse
import torch
import os
import copy

# parameter order follows the R names(): nu, k1..k7, Ka, Kb, m, n
def ode_model(t, y, p):
    nu, k1, k2, k3, k4, k5, k6, k7, Ka, Kb, m, n = p
    M, P, Pp = y
    hill = (k7 * P * Pp**n) / (Kb**n + Pp**n)
    dM = nu / (1 + (Pp / Ka)**m) - k1 * M
    dP = k2 * M - (k3 + k4) * P + k6 * Pp - hill
    dPp = k4 * P - (k5 + k6) * Pp + hill
    return [dM, dP, dPp]

def period_amp_estimate(x_pt_s, y0=(0.0, 0.0, 0.0), maxtime=520, spinoff=200):
    # 1. fine grid, step 0.2 (rounded to avoid float drift at the spinoff boundary)
    fine_t = np.round(np.linspace(1, maxtime, int(round((maxtime - 1) / 0.2)) + 1), 1)
    sol = solve_ivp(ode_model, (fine_t[0], fine_t[-1]), y0, args=(x_pt_s,),
                    t_eval=fine_t, method="LSODA", rtol=1e-6, atol=1e-6)
    time, M = sol.t, sol.y[0]

    # 2. drop the transient
    keep = time > spinoff
    time, M = time[keep], M[keep]

    # 3. rough period from the periodogram (fs = 5 samples per time unit)
    n = len(M)
    freqs, spec = periodogram(
        M, fs=5, window=get_window(("tukey", 0.2), n),  # ~ R's taper = 0.1 per side
        detrend="linear", nfft=next_fast_len(n, real=True), scaling="density",
    )
    freqs, spec = freqs[1:], spec[1:]                    # R's spectrum() has no zero frequency
    rough = 1 / freqs[np.argmax(spec)]

    # 4. refine with peak-to-peak timing
    anker = time[np.argmax(M)]
    local = (time > anker + rough - 5) & (time < anker + rough + 5)
    if not local.any():
        estimate = np.nan
    else:
        anker_2 = time[local][np.argmax(M[local])]
        estimate = anker_2 - anker

    # 5. reject non-oscillating cases
    if not np.ptp(M) > 0.03:
        estimate = np.nan

    # 6. amplitude
    amp = M.max() - M.min()
    return estimate, amp

def period_amp_estimate_robust(x_pt_s, y0=(0.0, 0.0, 0.0), maxtime=10000, spinoff=200, delta_t=0.2):
    # fine grid: 1, 1.2, ..., maxtime (rounded to avoid float drift at the spinoff boundary)
    n_grid = int(round((maxtime - 1) / delta_t)) + 1
    fine_t = np.round(1 + delta_t * np.arange(n_grid), 6)
    sol = solve_ivp(ode_model, (fine_t[0], fine_t[-1]), y0, args=(x_pt_s,),
                    t_eval=fine_t, method="LSODA", rtol=1e-6, atol=1e-6)
    if not sol.success or sol.y.shape[1] != n_grid:
        return np.nan, np.nan

    time, M = sol.t, sol.y[0]

    # drop the transient
    keep = time > spinoff
    time, M = time[keep], M[keep]
    n = len(M)

    # rough period from the periodogram (fs = 1/delta_t)
    freqs, spec = periodogram(
        M, fs=1 / delta_t, window=get_window(("tukey", 0.2), n),  # ~ R's 10% taper per side
        detrend="linear", nfft=next_fast_len(n, real=True), scaling="density",
    )
    freqs, spec = freqs[1:], spec[1:]          # R's spectrum() has no zero frequency
    rough = 1 / freqs[np.argmax(spec)]

    # first anchor: max of M within the middle 50% of the time range
    q1, q3 = np.quantile(time, [0.25, 0.75])   # default linear interpolation = R's type 7
    center = (time > q1) & (time < q3)
    anchor = time[center][np.argmax(M[center])]

    # second anchor: search window capped at 40% of the rough period
    window_size = min(5, rough * 0.4)
    local = (time > anchor + rough - window_size) & (time < anchor + rough + window_size)
    if not local.any():
        return np.nan, np.nan
    anchor_2 = time[local][np.argmax(M[local])]
    estimate = anchor_2 - anchor

    # dead-curve check on the last 10% of the series
    start = int(round(n * 0.9)) - 1            # R's round(n*0.9):n is 1-indexed
    y_final = M[start:]
    amp = y_final.max() - y_final.min()

    if amp <= 0.03:
        estimate = np.nan                      # dead curve; amplitude is still returned

    return estimate, amp


def main(args):
    if args.method == "MCMC":
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        Techno_thin = np.loadtxt(os.path.join(BASE_DIR, "Techno_thin.txt"))   # float64
        print(Techno_thin.shape, flush=True)                                  # expect (rows, 12)

        out_dir = os.path.join(BASE_DIR, "sensitivity_output")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"idx{args.theta_index}_amt{args.sensi_amount}_start{args.start}_maxtime{args.maxtime}"
        
        period_list, amp_list = [], []
        try:
            for j in range(1000 * args.start, 1000 * (args.start + 1)):
                input_theta = Techno_thin[j].copy()
                input_theta[args.theta_index] *= args.sensi_amount
                period, amp = period_amp_estimate_robust(input_theta, maxtime=args.maxtime)
                period_list.append(period)
                amp_list.append(amp)
                print(j, period, amp, flush=True)
        finally:
            np.savetxt(os.path.join(out_dir, f"period_{tag}.txt"), period_list)
            np.savetxt(os.path.join(out_dir, f"amp_{tag}.txt"), amp_list)

def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for each method.")
    parser.add_argument('--method', type=str, default='MCMC', help='MCMC or NPE')
    parser.add_argument('--start', type=int, default=0, help='start_index')
    parser.add_argument('--theta_index', type=int, default=1, help='theta_index: 1-9')
    parser.add_argument('--sensi_amount', type=float, default=1.0, help='Sensitivity amount 0.6, 0.8, 1.0, 1.2, 1.4')
    parser.add_argument('--maxtime', type=int, default=520, help='Maximum time for the analysis')
        
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
