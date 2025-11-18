
import os, sys, torch,pickle, argparse 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import observation_lists, Bounds, Posteriors
from sbibm.metrics.c2st import c2st


def run_similiarity(task, measure, x0_ind, seed, post_n_samples, num_training, cond):
    x0_list = observation_lists(task)
    x0 = x0_list[x0_ind]
    torch.manual_seed(seed)
    print(x0)

    limits = Bounds(task)
    posterior = Posteriors(task)
    task_benchmark = ["two_moons"]
    if task in task_benchmark:
        true_sample = posterior(j = x0_ind+1)
    else:
        true_sample = posterior(torch.tensor(x0), n_samples=post_n_samples, bounds=limits)
    
    output_file_path = f"../depot_hyun/hyun/NPE_ABC/nets/{task}/J_{int(num_training/1000)}K/{task}_{seed}_{cond}.pkl"    
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"NPE results file not found: {output_file_path}")
        
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    posterior = saved_data['posterior']
    x0 = torch.tensor(x0, dtype = torch.float32)
    if x0.ndim == 1:
        x0= torch.reshape(x0, (1, x0.size(0)))
    sample_post = posterior.sample((post_n_samples,), x=torch.tensor(x0))

    if measure == "c2st":
        sample_post_size = sample_post.size(0)
        dist = c2st(true_sample[:sample_post_size], sample_post[:sample_post_size])
    print("c2st: ", dist)  
    # Save
    output_dir = f"../depot_hyun/hyun/NPE_ABC/NPE_{measure}_results/{task}/J_{int(num_training/1000)}K"   
    os.makedirs(output_dir, exist_ok=True)
    torch.save(dist, os.path.join(output_dir, f"result_x0_{x0_ind}_seed_{seed}.pt"))  # Customize filename as needed
    

def get_args():
    parser = argparse.ArgumentParser(description="Run SLURM job for simulation.")
    parser.add_argument('--task', type=str, required=True, help='Task type')
    parser.add_argument('--measure', type=str, required=True, default = "c2st", help='Measurement type (c2st, SW)')
    parser.add_argument('--x0_ind', type=int, required=True, help='x0 index')
    parser.add_argument('--seed', type=int, required=True, help='seed num')
    parser.add_argument('--post_n_samples', type=int, default=10_000, help='Number of samples from posterior distributions')
    parser.add_argument("--num_training", type=int, default=500_000,
                        help="Number of simulations for training (default: 500_000)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    return parser.parse_args()

    
if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments    
    run_similiarity(args.task, args.measure, args.x0_ind, args.seed, args.post_n_samples, args.num_training, args.cond_den)


