import torch
import simulator
from simulator import Simulators, Priors, observation_lists, Bounds, true_Posteriors, simulator_slcp_distractors

def main():
    # Example usage of the Simulators class
    #simulators = simulator.Simulators(task="slcp")
    priors = simulator.Priors(task="slcp")
    
    # Sample theta from the prior
    theta = priors.sample((100,))  # Sample 5 parameter sets

    # Run the simulator
    X = simulator_slcp_distractors(theta)

    print("Sampled theta:")
    print(theta)
    print("Simulated data X:")
    print(X)    
    print(X.size())

if __name__ == "__main__":
    main()  # Pass the entire args object to the main function
