import sys, os
import torch
import numpy as np
from sbi.inference import NPE
import pickle
import os
import argparse
import time
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import Simulators, Priors, observation_lists, true_Posteriors
from utils.evaluate import create_c2st_job_script
from sbibm.metrics.c2st import c2st

def main(args):
    # Set the random seed
    torch.manual_seed(args.seed)

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    simulators = Simulators(args.task)

    # Sample theta from the prior
    theta = priors().sample((args.num_trainings,))

    # Run the simulator
    X = simulators(theta)

    # Load the inference object if it exists
    output_dir_tmp = f"../depot_hyun/NPE_nets/{args.task}/J_{int(args.num_trainings/1000)}K"
    
    # Create the temporal directory if it doesn't exist
    if not os.path.exists(output_dir_tmp):
        os.makedirs(output_dir_tmp)
        print(f"Temporal Directory '{output_dir_tmp}' created.")
    else:
        print(f"Temproal Directory '{output_dir_tmp}' already exists.")

    output_file_path_tmp = os.path.join(output_dir_tmp, f"{args.task}_{args.seed}_{args.cond_den}_tmp.pkl")

    inference = SNPE(density_estimator=args.cond_den)
    inference = inference.append_simulations(theta, X)  # Assumes theta and X are defined earlier
    each_epochs = 50

    # Check if we need to load a previous inference object
    start_time = time.time()  # Start timer
    if os.path.exists(output_file_path_tmp):
        with open(output_file_path_tmp, 'rb') as f:
            data = pickle.load(f)
            inference = data['inference']
            num_iter = data['num_iter']
            print("Loaded previous inference object.")
        torch.manual_seed(args.seed + num_iter)
        inference.train(max_num_epochs = each_epochs * (num_iter + 1), resume_training=True, force_first_round_loss=True)
        print(f"trained with epochs {inference.summary['epochs_trained'][-1]}")
        num_iter += 1

    else:
        inference.train(max_num_epochs=each_epochs)
        num_iter = 1
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    with open(output_file_path_tmp, 'wb') as f:
        pickle.dump({'inference': inference, 'elapsed_time': elapsed_time, 'num_iter': num_iter}, f)

    print(f"Saved inference object and elapsed time to '{output_file_path_tmp}'.")


    # Check if the training finished successfully
    if inference.summary['epochs_trained'][-1] == num_iter * each_epochs + 1:
        print(inference.summary['epochs_trained'][-1])
        print("Training not completed. Generating a new job script...")
        create_job_script(args)

    else:
        print("Training completed successfully.")
        output_dir = f"../depot_hyun/NPE_nets/{args.task}/J_{int(args.num_trainings/1000)}K"
        output_file_path = os.path.join(output_dir, f"{args.task}_{args.seed}_{args.cond_den}.pkl")
        with open(output_file_path, 'wb') as f:
            pickle.dump({'inference': inference, 'elapsed_time': elapsed_time, 'num_iter': num_iter}, f)

        os.remove(output_file_path_tmp)
        print(f"File '{output_file_path_tmp}' has been removed.")
        print(f"Learning done with epochs{inference.summary['epochs_trained'][-1]}")
        task_params = get_task_parameters(args.task)
        x0_list = task_params["x0_list"]
        for j in range(len(x0_list)):
            create_c2st_job_script(args.cond_den, "NPE", args.task, "c2st", 10000, args.num_trainings, j, args.seed)

    
def create_job_script(args):
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --output=NPE/NPE_{args.cond_den}/output_log/output_log_%a.log
#SBATCH --error=NPE/NPE_{args.cond_den}/output_log/error_log_%a.txt

# Load the required Python environment
module use /depot/wangxiao/etc/modulescreate_c2st_job_script
module load conda-env/sbi_pack-py3.11.7

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NeuralABC
cd $SLURM_SUBMIT_DIR
# Run the Python script for additional training
echo "Resuming training..."
python NPE/NPE_run2.py --task "{args.task}" --seed {args.seed} --cond_den "{args.cond_den}" --num_trainings {args.num_trainings}
echo "## Job completed for task {args.task} with seed {args.seed} ##"
"""
    # Create the directory if it doesn't exist
    output_dir = f"NPE/{args.task}/slurm_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    job_file_path = os.path.join(output_dir, f"{args.cond_den}_{args.seed}_continued.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created and submitted: {job_file_path}")

    # Submit the job immediately
    os.system(f"sbatch {job_file_path}")
    print(f"Job {job_file_path} submitted.")


def create_c2st_job_script(cond_den, method, task, measure, post_n_samples, num_trainings, j, i, use_gpu=False):
    sbatch_gpu_options = """
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
""" if use_gpu else ""

    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=statdept
#SBATCH -p cpu
#SBATCH -q standby
#SBATCH --output={measure}/{task}/output_log/output_log_%A.log
#SBATCH --error={measure}/{task}/error_log/error_log_%A.txt
{sbatch_gpu_options}

# Create the output_log directory if it doesn't exist
mkdir -p {measure}/{task}/output_log
mkdir -p {measure}/{task}/error_log

module load conda
conda activate /depot/wangxiao/apps/hyun18/sbi_pack

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

# Run the Python script for the current simulation
echo "Running simulation for task '{task}', '{method}_{cond_den}', {measure} j={j}, i={i}..."
python ./utils/get_measure.py --task {task} --num_training {num_trainings} --methods {method}_"{cond_den}" --post_n_samples {post_n_samples} --measure {measure} --x0_ind {j} --seed {i}
echo "## Job completed for task '{task}', '{cond_den}', {measure} j={j}, i={i}" ##"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"{measure}/{task}/slurm_files"
    os.makedirs(output_dir, exist_ok=True)

    job_file_path = os.path.join(output_dir, f"{task}_{cond_den}_c2st_j{j}_i{i}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")


def get_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run simulations and inference.")
    parser.add_argument('--task', type=str, required=True, help='Simulation type: twomoons, SLCP, or MoG')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--cond_den', type=str, required=True, help='Conditional density estimator type: mdn, maf, nsf')
    parser.add_argument('--method', type=str, required=True, help='NPE, NLE')
    parser.add_argument('--num_trainings', type=int, default=500_000, help='Number of simulations to run')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()  # Parse command-line arguments
    main(args)  # Pass the entire args object to the main function
