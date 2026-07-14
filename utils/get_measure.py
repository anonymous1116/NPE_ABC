
import os, sys, torch, pickle, argparse 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import observation_lists, Bounds, true_Posteriors, task_benchmark, Priors, Simulators
from sbibm.metrics.c2st import c2st
from pathlib import Path
from sbi.analysis import pairplot
import matplotlib.pyplot as plt
from sbi.inference import NPSE
import dill
def run_similiarity(task, measure, x0_ind, seed, post_n_samples, num_training, cond, method):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x0_list = observation_lists(task)
    x0 = x0_list[x0_ind]
    torch.manual_seed(seed)
    if args.task in ["table_dp_22"]:
        x0 = x0[:3]
    elif args.task in ["table_dp_33"]:
        x0 = x0[:8]
    elif args.task in ["table_dp_44"]:
        x0 = x0[:15]
    elif args.task in ["table_dp_55"]:
        x0 = x0[:24]
    elif args.task in ["table_dp_66"]:
        x0 = x0[:35]
    print(x0)
    

    limits = Bounds(task)
    posterior = true_Posteriors(task)
    #task_benchmark = ["two_moons", "bernoulli_glm2", "slcp_summary_transform2", "double_slcp_summary_transform2", "mog_10", "slcp_distractors"]
    if task in task_benchmark:
        true_sample = posterior(j = x0_ind+1)
    else:
        true_sample = posterior(torch.tensor(x0), n_samples=post_n_samples, bounds=limits)

    if method in ["FMPE", "NPSE"]:
        output_file_path = f"../depot_hyun/hyun/NPE_ABC/{method}_nets/{task}/J_{int(num_training/1000)}K/{task}_{seed}.pkl"    
    else:    
        output_file_path = f"../depot_hyun/hyun/NPE_ABC/nets/{task}/J_{int(num_training/1000)}K/{task}_{seed}_{cond}.pkl"    
        
    
    x0 = torch.tensor(x0, dtype = torch.float32)
    if x0.ndim == 1:
        x0= torch.reshape(x0, (1, x0.size(0)))


    if method == "NPSE":
        # Step 1: rebuild structure with dummy training
        simulators = Simulators(task)
        priors = Priors(task)
        inference = NPSE(prior=priors)
        theta_tmp = priors.sample((10,))
        X_tmp = simulators(theta_tmp)
        inference.append_simulations(theta_tmp, X_tmp)
        density_estimator = inference.train(max_num_epochs=1)

        # Step 2: load saved weights
        saved_data = torch.load(output_file_path.replace('.pkl', '.pt'), map_location='cpu')
        density_estimator.net.load_state_dict(saved_data['net_state_dict'])
        density_estimator._embedding_net.load_state_dict(saved_data['embedding_net_state_dict'])
        density_estimator = density_estimator.eval()

        # Step 3: build posterior
        posterior = inference.build_posterior(vector_field_estimator=density_estimator)
        sample_post = posterior.sample((10_000,), x=x0)
    else:
        with open(output_file_path, 'rb') as f:
            saved_data = pickle.load(f)
    
        posterior = saved_data['posterior']
        sample_post = posterior.sample((post_n_samples,), x=torch.tensor(x0))

    if measure == "c2st":
        sample_post_size = sample_post.size(0)
        dist = c2st(true_sample[:sample_post_size], sample_post[:sample_post_size])
    print("c2st: ", dist)  
    # Save
    
    output_dir = f"../depot_hyun/hyun/NPE_ABC/{method}_{measure}_results/{task}/J_{int(num_training/1000)}K"   
    os.makedirs(output_dir, exist_ok=True)
    torch.save(dist, os.path.join(output_dir, f"result_x0_{x0_ind}_seed_{seed}.pt"))  # Customize filename as needed

    # Save to output_dir
    pairplot(true_sample, figsize=(6,6), limits = limits)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()

    pairplot(sample_post[:sample_post_size], figsize=(6,6), limits = limits)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
def get_args():
    parser = argparse.ArgumentParser(description="Run SLURM job for simulation.")
    parser.add_argument('--task', type=str, required=True, help='Task type')
    parser.add_argument('--measure', type=str, default = "c2st", help='Measurement type (c2st, SW)')
    parser.add_argument('--x0_ind', type=int, required=True, help='x0 index')
    parser.add_argument('--seed', type=int, required=True, help='seed num')
    parser.add_argument('--post_n_samples', type=int, default=10_000, help='Number of samples from posterior distributions')
    parser.add_argument("--num_training", type=int, default=500_000,
                        help="Number of simulations for training (default: 500_000)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--method', type=str, default='NPE', 
                        help='method_type: NPE, FMPE, NPSE')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments    
    run_similiarity(args.task, args.measure, args.x0_ind, args.seed, args.post_n_samples, args.num_training, args.cond_den, args.method)


#python utils/get_measure.py --task "bernoulli_glm2" --measure "c2st" --x0_ind 1 --seed 1 --post_n_samples 10000 --num_training 1000 --method "NPSE"