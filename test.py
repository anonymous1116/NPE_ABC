import torch
import simulator
import time 
from simulator import Simulators, Priors, observation_lists, Bounds, true_Posteriors, simulator_slcp_distractors

def main():
    # Example usage of the Simulators class
    #simulators = simulator.Simulators(task="slcp")
    priors = simulator.Priors("slcp_summary_transform2")
    
    start_time = time.time()
    # Sample theta from the prior
    num = 1_000_000
    theta = priors.sample((num,))  # Sample 5 parameter sets

    # Run the simulator
    X = simulator_slcp_distractors(theta)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulated {int(num/1000)}K samples in {elapsed_time/60:.2f} minutes") 
    #        GPU: ,           CPU:  for 1M samples
    # 10K   0.02 min.     0.11 min
    #100K   0.11 min.     1.01 min
    # 1M    0.11 min.     10.1 min
    print("Simulated data X:")
    print(X.size())

if __name__ == "__main__":
    main()  # Pass the entire args object to the main function
