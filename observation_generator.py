import torch, argparse, sys, random
import numpy as np
import os, pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, observation_lists

def main(args):
    if args.task == "my_ten_twomoons":
        random.seed(2826)
        torch.manual_seed(2826)
        x0_list = []
        for j in range(1):
            sample = np.random.choice(np.arange(0, 10), size=10, replace=True)
            tmp = observation_lists("my_twomoons")
            tmp = np.array(tmp)
            x0_list.append(np.concatenate(tmp[sample],0).tolist())
        x0_list = torch.tensor(x0_list, dtype = torch.float32)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(x0_list, f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_ten_twomoons_obs.pt")    
        print(x0_list)
        
def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, default='twomoons', help='Simulation type: twomoons, MoG, Lapl, GL_U or SLCP')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
