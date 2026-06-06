import sys, os
import torch
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import observation_lists, Bounds, true_Posteriors, task_benchmark
from sbibm.metrics.c2st import c2st


def main(args):
    x0_list = observation_lists(args.task)
    limits = Bounds(args.task)
    posterior_fn = true_Posteriors(args.task)

    samples_dir = (
        f"../depot_hyun/hyun/NPE_ABC/NPSE_nets_NABC/NPSE_samples_NABC/{args.task}"
        f"/J_{int(args.num_training/1000)}K"
    )
    samples_path = os.path.join(samples_dir, f"samples_x0_{args.obs_idx}_seed_{args.seed}.pt")
    time_path = os.path.join(samples_dir, f"samples_x0_{args.obs_idx}_seed_{args.seed}_time.pt")

    sample_post = torch.load(samples_path)
    elapsed_time = torch.load(time_path)

    x_o = torch.tensor(x0_list[args.obs_idx], dtype=torch.float32)

    if args.task in task_benchmark:
        true_sample = posterior_fn(j=args.obs_idx + 1)
    else:
        true_sample = posterior_fn(torch.tensor(x_o), n_samples=10_000, bounds=limits)

    dist = c2st(true_sample, sample_post)
    print(f"c2st: {dist}", flush=True)

    # Save results
    output_dir = (
        f"../depot_hyun/hyun/NPE_ABC/NPSE_c2st_NABC_results/{args.task}"
        f"/J_{int(args.num_training/1000)}K"
    )
    os.makedirs(output_dir, exist_ok=True)
    torch.save(dist, os.path.join(output_dir, f"result_x0_{args.obs_idx}_seed_{args.seed}.pt"))
    torch.save(elapsed_time, os.path.join(output_dir, f"result_x0_{args.obs_idx}_seed_{args.seed}_time.pt"))

    # Delete temp sample files
    os.remove(samples_path)
    os.remove(time_path)
    print(f"Deleted temp files for x0={args.obs_idx}, seed={args.seed}", flush=True)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate NPSE posterior samples with C2ST.")
    parser.add_argument('--task', type=str, default='twomoons', help='Task name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_training', type=int, default=100_000, help='Number of training simulations')
    parser.add_argument('--obs_idx', type=int, default=0, help='Index into observation_lists')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())