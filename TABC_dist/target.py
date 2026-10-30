import torch
import numpy as np
import os, sys, pickle
import argparse
import sbibm
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sbi.analysis import pairplot
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sbibm.metrics.c2st import c2st
from sbibm.metrics.mmd import mmd
from simulator import Priors, Simulators, Bounds, observation_lists, true_Posteriors, task_benchmark
from help_functions import UnifSample, param_box, truncated_mvn_sample, ABC_rej2
from utils.get_measure import sliced_wasserstein

def main(args):
    seed = args.seed
    #torch.set_default_device("cpu")
    
    original_task = "bernoulli_glm2" if args.task.startswith("bernoulli_glm2") else args.task

    current_dir = os.path.dirname(os.path.abspath(__file__))
    disperse_num = int(args.task.split("_")[-1][8:])
            
    A = torch.load(f"{current_dir}/../utils/files/rotation_A_{disperse_num}x10_seed0.pt")
            
    torch.manual_seed(seed)
    np.random.seed(seed)

    L = args.L
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    priors = Priors(original_task)
    simulators = Simulators(original_task)
    bounds = Bounds(original_task)
    true_posteriors = true_Posteriors(original_task)
        


    chunk_size = 5_000_000
    num_chunks = L // chunk_size
    
    start_time = time.time()
    x0 = observation_lists(original_task)[args.x0_ind]

    print(x0)
    if x0.ndim == 1:
        x0 = torch.reshape(x0, (1,x0.size(0)))
        
    chunk_size_cal = 10_000
    print("x0_size", x0.size(), flush = True)
    #print("X_cal size", X_cal.size(), flush = True)
    
    Y_cal = priors.sample((1_000_000,))
    X_cal = simulators(Y_cal)

    output_file_path = os.path.join(f'{current_dir}/../depot_hyun/hyun/NPE_ABC/nets/{args.task}/J_{int(args.num_training/1000)}K/{args.task}_{seed}_{args.cond_den}.pkl')
    with open(output_file_path, 'rb') as f:
        saved_data = pickle.load(f)
    density_estimator_npe = saved_data["density_estimator"]
    density_estimator_npe_gpu = density_estimator_npe.to(device).eval()

    flow = density_estimator_npe_gpu.net
    transform=flow._transform
    embed = flow._embedding_net
    
    index_ABC = ABC_rej2(x0, X_cal, 1e-2, device)
    X_cal, Y_cal = X_cal[index_ABC], Y_cal[index_ABC]    


    x0_transform = torch.matmul(x0.to(device), A.T.to(device))
    X_cal_transform = torch.matmul(X_cal.to(device), A.T.to(device))
    
    with torch.no_grad():
        tmp, _ = transform.forward(Y_cal.to(device), context = embed(X_cal_transform.to(device)) )
        adj, _ = transform.inverse(tmp, context = embed(x0_transform.expand((tmp.size(0),x0_transform.size(1))).to(device)))    
    adj = adj.cpu()

    X_abc = []
    Y_abc = []
    
    true_sample = true_posteriors(j = args.x0_ind+1)
    
    if bounds is not None:
        adj = torch.clamp(adj, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])

    with torch.no_grad():
        max_vals = torch.max(adj,0).values
        min_vals = torch.min(adj,0).values
    
    priors_mean = torch.zeros(10)
    priors_std = torch.ones(10) * np.sqrt(2)

    print("max_vals:", max_vals)   
    print("min_vals:", min_vals)


    print(X_cal.size())
    new_tol = 1e-2
    for i in range(num_chunks + 1): 
        start = i * chunk_size
        end = (i + 1) * chunk_size if (i + 1) * chunk_size < L else L
        nums = end-start

        if nums == 0:
            break
        if args.task.startswith("bernoulli_glm2"):
            Y_chunk = truncated_mvn_sample(nums, priors_mean, priors_std, min_vals, max_vals)
        else:
            Y_chunk = param_box(UnifSample(bins = 10), adj, num=nums)
        
        X_chunk = simulators(Y_chunk)

        index_ABC = ABC_rej2(x0, X_chunk, args.tol*(1/new_tol), device)
    
        X_abc.append(X_chunk[index_ABC])
        Y_abc.append(Y_chunk[index_ABC])
        print(f"{i}th iteration out of {num_chunks}", flush = True)
        del X_chunk, Y_chunk, index_ABC
    
    X_abc = torch.cat(X_abc)
    Y_abc = torch.cat(Y_abc)
    
    index_ABC = ABC_rej2(x0, X_abc, new_tol, device)
    X_abc_tmp, Y_abc_tmp = X_abc[index_ABC], Y_abc[index_ABC]

    X_abc_tmp_transform = torch.matmul(X_abc_tmp.to(device), A.T.to(device))
            


    print("X_abc size", X_abc.size())
    with torch.no_grad():
        tmp, _ =  transform.forward(Y_abc_tmp.to(device), context = embed(X_abc_tmp_transform.to(device)) )
        new_theta, _ = transform.inverse(tmp, context = embed(x0_transform.expand((tmp.size(0),x0_transform.size(1))).to(device)))    
    
    if bounds is not None:
        new_theta = torch.clamp(new_theta, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])
        tol_bound = 10000/new_theta.size(0) * new_tol
        print("new_tol", new_tol, "tol_bound:", tol_bound)
        index_ABC = ABC_rej2(x0, X_abc, tol_bound, device)
        X_abc, Y_abc = X_abc[index_ABC], Y_abc[index_ABC]
        X_abc_transform = torch.matmul(X_abc.to(device), A.T.to(device))
            
        with torch.no_grad():
            tmp, _ =  transform.forward(Y_abc.to(device), context = embed(X_abc_transform.to(device)) )
            new_theta, _ = transform.inverse(tmp, context = embed(x0_transform.expand((tmp.size(0),x0_transform.size(1))).to(device)))    
        new_theta = torch.clamp(new_theta, min = torch.tensor(bounds)[:,0], max = torch.tensor(bounds)[:,1])
        
    new_theta = new_theta.cpu()
    print("new_theta:", new_theta)
    print("new_theta_max:", torch.max(new_theta,0).values)
    print("new_theta_min:", torch.min(new_theta,0).values)
    # 4) Now call your fast function (or sbi’s sample_batched) on GPU
    end_time = time.time()

    elapsed_time = end_time - start_time  # Calculate elapsed time
    
    print("TABC sample size: ", new_theta.size())
    results_size = min(10_000, new_theta.size(0))



    dist0 = c2st(true_sample[:results_size], new_theta[:results_size])
    dist1 = mmd(true_sample[:results_size], new_theta[:results_size])
    dist2 = sliced_wasserstein(true_sample[:results_size], new_theta[:results_size])
    dist3 = sliced_wasserstein(true_sample[:results_size], new_theta[:results_size], p=1)
    
    print(tmp)    
    
    sci_str = format(args.tol, ".0e")
    print(sci_str)  # Output: '1e-02'
    
    output_dir = f"../depot_hyun/hyun/NPE_ABC/Tdist/{args.task}/target/J_{int(args.num_training/1000)}K/{int(args.L/1_000_000)}M_eta{sci_str}"
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")    

    # Save to output_dir
    pairplot(true_sample, figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference_bounds.png")
    plt.close()

    pairplot(true_sample, figsize=(6,6))
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_reference.png")
    plt.close()
    
    pairplot(new_theta[:10000], figsize=(6,6), limits = bounds)
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated_bounds.png")
    plt.close()
    
    pairplot(new_theta[:10000], figsize=(6,6))
    plt.savefig(Path(output_dir) / f"x0{args.x0_ind}_seed{args.seed}_calibrated.png")
    plt.close()
    
    
    device_name = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    
    torch.save({'c2st': dist0, 'mmd': dist1, 'SW1': dist3, 'SW2': dist2, 'device': device_name, 'elapsed_time': elapsed_time}, 
               os.path.join(output_dir, f"result_x0_{args.x0_ind}_seed_{args.seed}.pt"))  
    
    #
def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--L", type = int, default = 10_000_000,
                        help = "Calibration data size (default: 10M)")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: Lapl, MoG')
    parser.add_argument("--num_training", type=int, default=100_000, 
                        help="Number of training data of NPE (default: 100_000)")
    parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance value for ABC (any positive float, default: 1e-4 but less than 1e-2)")
    parser.add_argument('--cond_den', type=str, default='nsf', 
                        help='Conditional density estimator type: mdn, maf, nsf')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"x0_ind: {args.x0_ind}")
    print(f"seed: {args.seed}")
    print(f"L: {args.L}")
    print(f"task: {args.task}")
    print(f"num_training: {args.num_training}")
    print(f"tol: {args.tol}")
    print(f"cond_den: {args.cond_den}")


#python ABC_calibration/calibrating_flow_latent_target.py --x0_ind 1 --seed 1 --task "my_five_twomoons_err90" --num_training 5000000 --L 1000000 --tol 1e-2
