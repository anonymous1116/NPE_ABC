import torch, argparse, sys, random
import numpy as np
import os, pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Priors, Simulators, observation_lists, MoG_posterior, Bounds, true_Posteriors

def main(args):
    if args.task == "my_ten_twomoons":
        random.seed(2826)
        torch.manual_seed(2826)
        x0_list = []
        for j in range(10):
            sample = np.random.choice(np.arange(0, 10), size=10, replace=True)
            tmp = observation_lists("my_twomoons")
            tmp = np.array(tmp)
            x0_list.append(np.concatenate(tmp[sample],0).tolist())
        x0_list = torch.tensor(x0_list, dtype = torch.float32)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(x0_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_ten_twomoons_obs.pt")    
        print(x0_list)
    
    elif args.task == "my_fifty_twomoons":
        random.seed(2826)
        torch.manual_seed(2826)
        x0_list = []
        theta_obs = torch.rand((10,100)) * 8 - 4 #support: [-4,4]
        X_obs = Simulators("my_fifty_twomoons")(theta_obs)
        true_posterior = true_Posteriors("my_twomoons")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        torch.save(X_obs, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_fifty_twomoons_obs.pt")
        for j in range(10):
            tmp =X_obs[j]
            post = []
            torch.manual_seed(2826+j)
        
            for k in range(50):
                tmp2 = tmp[2*k:(2*k+2)]
                two_moons_post = true_posterior(torch.tensor(tmp2)[None, :], n_samples=10_000, bounds=Bounds("my_twomoons"))
                post.append(two_moons_post)
            post_sample = torch.cat(post, dim = 1)
            torch.save(post_sample, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_post_{j+1}.pt")
            print(f"{j}th obs: {tmp}")
            

    elif args.task.startswith("my_five_twomoons"):
        #permunation
        if args.task == "my_five_twomoons_err10":
            noise_num = 10
        elif args.task == "my_five_twomoons_err30":
            noise_num = 30
        elif args.task == "my_five_twomoons_err50":
            noise_num = 50
        elif args.task == "my_five_twomoons_err70":
            noise_num = 70
        elif args.task == "my_five_twomoons_err90": 
            noise_num = 90
        elif args.task == "my_five_twomoons":
            noise_num = 0
        else:
            raise ValueError("Invalid task name for error level.")
        
        torch.manual_seed(2825)
        permute = torch.randperm(10+noise_num)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if noise_num > 0:
            torch.save(permute, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_permutation.pt")
        
        random.seed(2826)
        torch.manual_seed(2826)
        # Posteriors
        bounds = Bounds("my_five_twomoons")
        true_posterior = true_Posteriors("my_five_twomoons")
        
        x0_list = []
        #theta_obs = Priors("my_five_twomoons").sample((10,))
        theta_obs = torch.rand((10,10)) * 8 - 4 #support: [-4,4]
        
        X_obs = Simulators("my_five_twomoons")(theta_obs)
        
        if args.task == "my_five_twomoons":
            torch.save(X_obs, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_obs.pt")
            for j in range(10):
                tmp = X_obs[j]
                post_sample = true_posterior(torch.tensor(tmp)[None, :], n_samples=10_000, bounds=bounds)
                torch.save(post_sample, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_post_{j+1}.pt")
                

        else:
            for j in range(10):
                noise = torch.randn( (noise_num,)) * 2.0 
                tmp = X_obs[j]

                post_sample = true_posterior(torch.tensor(tmp)[None, :], n_samples=10_000, bounds=bounds)
                torch.save(post_sample, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_post_{j+1}.pt")
                tmp = torch.cat([tmp, noise])
                print(tmp)
                x0_list.append(tmp[permute].tolist())

            x0_list = torch.tensor(x0_list, dtype = torch.float32)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            torch.save(x0_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_obs.pt")    
            print(x0_list)
        

    elif args.task == "mog_10":
        random.seed(2826)
        torch.manual_seed(2826)
        x0_list = []
        for j in range(10):
            obs  = torch.rand(10) * 20 - 10
            x0_list.append(obs.tolist())
        x0_list = torch.tensor(x0_list, dtype = torch.float32)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(x0_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/mog_10_obs.pt")    
        print(x0_list)

        # Posteriors
        bounds = Bounds("mog_10")
        for j in range(10):
            x0 = x0_list[j][None, :]
            post_sample = MoG_posterior(x0, n_samples=10_000, bounds=bounds)
            torch.save(post_sample, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/mog_10_post_{j+1}.pt")
            print(post_sample)
    
    elif args.task in ["bernoulli_glm2_err10", "bernoulli_glm2_err30", "bernoulli_glm2_err50", "bernoulli_glm2_err70", "bernoulli_glm2_err90"]:
        #permunation
        torch.manual_seed(2825)
        if args.task == "bernoulli_glm2_err10":
            noise_num = 10
        elif args.task == "bernoulli_glm2_err30":
            noise_num = 30
        elif args.task == "bernoulli_glm2_err50":
            noise_num = 50
        elif args.task == "bernoulli_glm2_err70":
            noise_num = 70
        elif args.task == "bernoulli_glm2_err90":
            noise_num = 90
        else:
            raise ValueError("Invalid task name for error level.")
        
        permute = torch.randperm(noise_num+10)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        torch.save(permute, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_permutation.pt")
        
        random.seed(2826)
        torch.manual_seed(2826)
        # Posteriors
        true_posterior = true_Posteriors("bernoulli_glm2")
        
        x0_list = []
        for j in range(10):
            noise = torch.randn( (noise_num,)) * 2.0 
            tmp = observation_lists("bernoulli_glm2")[j]

            post_sample = true_posterior(j = j+1)
            torch.save(post_sample, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_post_{j+1}.pt")
            tmp = torch.cat([tmp, noise])
            print(tmp)
            x0_list.append(tmp[permute].tolist())

        x0_list = torch.tensor(x0_list, dtype = torch.float32)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        torch.save(x0_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_obs.pt")    
        print(x0_list)

    elif args.task in ["table_dp_22", "table_dp_33", "table_dp_44", "table_dp_55"]:
        import warnings
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        random.seed(2826)
        torch.manual_seed(2826)
        theta = Priors(args.task).sample((10000,))
        X = Simulators(args.task)(theta)

        # Keep only rows where all elements > 30
        mask = (X > 30).all(dim=1)
        theta_filtered = theta[mask]
        X_filtered = X[mask]

        n_filtered = mask.sum().item()
        print(f"Kept {n_filtered} / {len(mask)} observations")

        # Take 10 random samples
        n_select = 10
        if n_filtered < n_select:
            warnings.warn(f"Only {n_filtered} observations available, fewer than {n_select} requested.")
            idx = torch.randperm(n_filtered)
        else:
            idx = torch.randperm(n_filtered)[:n_select]

        X_selected = X_filtered[idx]
        torch.save(X_selected, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{args.task}_obs.pt")  
    else:
        print("Task not recognized.")



def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='my_ten_twomoons', help='Simulation type: my_ten_twomoons')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
