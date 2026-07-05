import torch
import os, sys
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Dirichlet, Multinomial
from sbi.utils import BoxUniform
import sbibm
import pyro.distributions as dist

# Optional: you can use this from torch.distributions if available
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from help_functions import SLCP_summary_transform2

def Bounds(task_name: str):
    task_name = task_name.lower()
    if task_name.startswith("bernoulli_glm2"):
        return None
    elif task_name in ["two_moons"]:
        return [[-1, 1]] * 2
    elif task_name in ["my_twomoons"]:
        return [[-5, 5]] * 2
    elif task_name.startswith("my_five_twomoons"):
        return [[-5, 5]] * 10
    elif task_name in ["my_ten_twomoons"]:
        return [[-5,5]] * 20
    elif task_name in ["my_fifty_twomoons"]:
        return [[-5,5]] * 100
    elif task_name in ["slcp_summary_transform2", "slcp_distractors", "slcp"]:    
        return [[-3, 3]] * 5
    elif task_name in ["double_slcp_summary_transform2"]:
        return [[-3, 3]] * 10
    elif task_name in ["mog_10"]:
        return [[-10, 10]] * 10
    elif task_name in ["mog_2_nabc"]:
        return [[-10, 10]] * 2
    elif task_name in ["mog_5_nabc"]:
        return [[-10, 10]] * 10
    elif task_name in ["mog_10_nabc"]:
        return [[-10, 10]] * 10
    elif task_name in ["table_dp_22"]:
        return [[0,1]] * 3
    else:
        raise ValueError(f"Unknown task name for bounds: {task_name}")

def Priors(task_name: str):
    task_name = task_name.lower()
    if task_name.startswith("bernoulli_glm2"):
        dim = 10
        loc = torch.zeros(dim)
        precision_diag = 0.5 * torch.ones(dim)
        precision_matrix = torch.diag(precision_diag)
        return MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
    elif task_name in ["two_moons"]:
        return BoxUniform(low = -1*torch.ones(2), high = 1*torch.ones(2))
    elif task_name in ["my_twomoons"]:
        return BoxUniform(low = -5*torch.ones(2), high = 5*torch.ones(2))
    elif task_name.startswith("my_five_twomoons"):
        return BoxUniform(low = -5*torch.ones(10), high = 5*torch.ones(10))
    elif task_name.startswith("my_fifty_twomoons"):
        return BoxUniform(low = -5*torch.ones(100), high = 5*torch.ones(100))
    elif task_name in ["my_ten_twomoons"]:
        return BoxUniform(low = -5*torch.ones(20), high = 5*torch.ones(20))
    elif task_name in ["mog_10"]:
        return BoxUniform(low = -10*torch.ones(10), high = 10*torch.ones(10))
    elif task_name in ["mog_2_nabc"]:
        return BoxUniform(low = -10*torch.ones(2), high = 10*torch.ones(2))
    elif task_name in ["mog_5_nabc"]:
        return BoxUniform(low = -10*torch.ones(5), high = 10*torch.ones(5))
    elif task_name in ["mog_10_nabc"]:
        return BoxUniform(low = -10*torch.ones(10), high = 10*torch.ones(10))
    
    elif task_name in ["slcp_summary_transform2", "slcp_distractors", "slcp"]:
        return BoxUniform(low = -3*torch.ones(5), high = 3*torch.ones(5))
    elif task_name in ["double_slcp_summary_transform2"]:
        return BoxUniform(low = -3*torch.ones(10), high = 3*torch.ones(10))
    elif task_name in ["table_dp_22"]:
        return Dirichlet(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    
    else:
        raise ValueError(f"Unknown task name for prior: {task_name}")

task_benchmark = ["two_moons", 
                  "bernoulli_glm2", "bernoulli_glm2_err10", "bernoulli_glm2_err30", "bernoulli_glm2_err50", "bernoulli_glm2_err70", "bernoulli_glm2_err90",
                  "slcp_summary_transform2", "double_slcp_summary_transform2", "mog_10", "slcp_distractors", "slcp", 
                  "my_five_twomoons_err10",
                  "my_five_twomoons_err30",
                  "my_five_twomoons_err50",
                  "my_five_twomoons_err70",
                  "my_five_twomoons_err90",
                  "mog_2_nabc", "mog_5_nabc", "mog_10_nabc",
                  "my_fifty_twomoons", "table_dp_22"]
    
class true_Posteriors:
    def __init__(self, task):
        self.task = task

    def __call__(self, obs=None, n_samples=100, bounds=None, **kwargs):
        # Handle the case where task is 'slcp' differently
        if self.task == "two_moons":
            return self.two_moons(kwargs.get('j', 0))
        elif self.task.startswith("bernoulli_glm2"):
            return self.bernoulli_glm2(kwargs.get('j', 0))
        elif self.task in ["slcp_summary_transform2", "slcp_distractors", "slcp"]:
            return self.slcp(kwargs.get('j', 0))
        elif self.task in ["double_slcp_summary_transform2"]:
            return self.double_slcp(kwargs.get('j', 0))
        elif self.task in ["my_five_twomoons_err10"]:
            return self.my_five_twomoons_err10(kwargs.get('j', 0))
        elif self.task in ["my_five_twomoons_err30"]:
            return self.my_five_twomoons_err30(kwargs.get('j', 0))
        elif self.task in ["my_five_twomoons_err50"]:
            return self.my_five_twomoons_err50(kwargs.get('j', 0))
        elif self.task in ["my_five_twomoons_err70"]:
            return self.my_five_twomoons_err70(kwargs.get('j', 0))
        elif self.task in ["my_five_twomoons_err90"]:
            return self.my_five_twomoons_err90(kwargs.get('j', 0))
        elif self.task in ["my_fifty_twomoons"]:
            return self.my_fifty_twomoons(kwargs.get('j', 0))

        elif self.task in ["mog_10"]:
            return self.mog_10(kwargs.get('j', 0))
        elif self.task in ["mog_2_nabc"]:
            return self.mog_2_nabc(kwargs.get('j', 0))
        elif self.task in ["mog_5_nabc"]:
            return self.mog_5_nabc(kwargs.get('j', 0))
        elif self.task in ["mog_10_nabc"]:
            return self.mog_10_nabc(kwargs.get('j', 0))
        elif self.task in ["table_dp_22"]:
            return self.table_dp_22(kwargs.get('j', 0))

        elif self.task in ["my_twomoons"]:
            return self.my_twomoons(obs, n_samples, bounds)
        elif self.task in ["my_five_twomoons", "my_five_twomoons_err2", "my_five_twomoons_err5", "my_five_twomoons_err10"]:    
            return self.my_five_twomoons(obs, n_samples, bounds)
    
        else:
            raise ValueError(f"Unknown task: {self.task}")
    def apply_bounds(self, samples, bounds):
        # Apply bounds to filter the samples
        if bounds is not None:
            index = []
            for j in range(samples.size()[1]):  # Iterate over each dimension
                ind = (samples[:, j] < bounds[j][1]) & (samples[:, j] > bounds[j][0])
                index.append(ind)
            index = torch.stack(index, 1)
            index = torch.all(index, 1)  # Check if all conditions hold per sample
            samples = samples[index]
        return samples
    
    def my_twomoons(self,obs, n_samples, bounds = None):
        if obs.ndim == 2:
            obs = obs.flatten()

        obs = torch.as_tensor(obs)
        c = 1 / np.sqrt(2)

        # Generate all samples at once
        p = Simulators("my_twomoons")(torch.zeros(100*n_samples, 2))

        # Vectorized q construction
        q0 = p[:, 0] - obs[0]
        q1 = obs[1] - p[:, 1]

        # Random sign flip
        signs = torch.where(
            torch.rand(100*n_samples) < 0.5,
            -1.0,
            1.0,
        )

        q0 = q0 * signs

        # Vectorized theta computation
        theta = torch.empty(100*n_samples, 2)
        theta[:, 0] = c * (q0 - q1)
        theta[:, 1] = c * (q0 + q1)

        if bounds is not None:
            theta = torch.clone(self.apply_bounds(theta, bounds))
        sam_ind = np.random.choice(np.arange(0, theta.size()[0]), n_samples, replace = False)
        
        return theta[sam_ind,:]

    def my_five_twomoons(self, obs, n_samples, bounds = None):
        if obs.ndim == 2:
            obs = obs.flatten()
        posterior = []
        for i in range(5):
            obs_tmp = obs[2*i: (2*i +2)]
            bounds_tmp = None if bounds is None else bounds[2*i: (2*i + 2)]
            tmp2 = self.my_twomoons(obs = obs_tmp, n_samples = n_samples, bounds = bounds_tmp)
            posterior.append(tmp2)
        posterior = torch.cat(posterior, dim = 1)
        return posterior

    def MoG(self,obs, n_samples, bounds = None):
        obs = torch.tensor(obs)
        if obs.ndim == 1:
            obs = torch.reshape(obs, (1, obs.size(0)))
        scale = [1.0, 0.1]
        n_samples2 = n_samples * 1000

        idx =  D.Bernoulli(torch.tensor(1/2)).sample((n_samples2,obs.size(1) )) 
        idx2 = 1 - idx

        tmp1 = D.Normal(obs[0], torch.tensor(scale[0])).sample((n_samples2,))
        tmp2 = D.Normal(obs[0], torch.tensor(scale[1])).sample((n_samples2,))

        tmp = tmp1 * idx + tmp2 * idx2
        if bounds is not None:
            tmp = torch.clone(apply_bounds(tmp, bounds))
        sam_ind = np.random.choice(np.arange(0, tmp.size()[0]), n_samples, replace = True)
        return tmp[sam_ind,:]


    def two_moons(self, j):
        task = sbibm.get_task("two_moons")
        return task.get_reference_posterior_samples(num_observation=j)

    def gaussian_mixture(self, j):
        task = sbibm.get_task("gaussian_mixture")  # See sbibm.get_available_tasks() for all tasks
        return task.get_reference_posterior_samples(num_observation=j)
    
    def gaussian_linear_uniform(self, j):
        task = sbibm.get_task("gaussian_linear_uniform")  # See sbibm.get_available_tasks() for all tasks
        return task.get_reference_posterior_samples(num_observation=j)
    
    def my_five_twomoons_err10(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err10_post_{j}.pt")    
        return post_sample
    def my_five_twomoons_err30(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err30_post_{j}.pt")    
        return post_sample
    def my_five_twomoons_err50(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err50_post_{j}.pt")    
        return post_sample
    def my_five_twomoons_err70(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err70_post_{j}.pt")    
        return post_sample
    def my_five_twomoons_err90(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err90_post_{j}.pt")    
        return post_sample
    
    def my_fifty_twomoons(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/my_fifty_twomoons_post_{j}.pt")    
        return post_sample

    def table_dp_22(self, j):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        post_sample = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/table_dp_22_post_{j}.pt")    
        return post_sample


    def slcp(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/slcp_benchmark/benchmark_post_sample_x0_{j}.pt")
            post_sample = torch.load(file_path)
            if post_sample.size(0) >12000:
                burn_in = int(post_sample.size(0) * 0.2)
                sam_ind = np.random.choice(np.arange(burn_in, post_sample.size(0)), 10_000, replace = False)
                post_sample = post_sample[sam_ind,:]
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    
    def bernoulli_glm2(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/bernoulli_glm/post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    
    def double_slcp(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/hyun/NPE_ABC/seeds/double_slcp_summary_transform2_post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    def mog_10(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/hyun/NPE_ABC/seeds/mog_10_post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    
    def mog_2_nabc(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/MoG_2/post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    def mog_5_nabc(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/MoG_5/post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    def mog_10_nabc(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/MoG_10/post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    

def observation_lists(task_name:str):
    task_name = task_name.lower()
    if task_name in ["two_moons"]:
        obs_list = []
        for j in range(1, 11):
            task = sbibm.get_task(task_name)
            observation = task.get_observation(num_observation=j)  # 10 per task
            obs_list.append(observation[0].tolist())
        return torch.tensor(obs_list)
    
    elif task_name in ["bernoulli_glm2"]:
        obs_list = []
        for j in range(1, 11):
            task = sbibm.get_task("bernoulli_glm")
            observation = task.get_observation(num_observation=j)  # 10 per task
            obs_list.append(observation[0].tolist())
        return torch.tensor(obs_list)
    
    elif task_name in ["my_twomoons"]:
        return torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.1, 0.1], 
                             [-0.1, -0.1], [-3.0, 3.0], [-2.0, 3.0], 
                             [-1.0, 1.0], [-0.5, 1.0], [-0.25, 0.5]], 
                             dtype = torch.float32)
    
    elif task_name in ["my_five_twomoons", "my_five_twomoons_err2", "my_five_twomoons_err5", 
                       "my_five_twomoons_err10", "my_five_twomoons_err30", "my_five_twomoons_err50",
                          "my_five_twomoons_err70", "my_five_twomoons_err90", 
                       "bernoulli_glm2_err10", "bernoulli_glm2_err30", "bernoulli_glm2_err50", "bernoulli_glm2_err70", "bernoulli_glm2_err90", "my_fifty_twomoons",
                       "table_dp_22"]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        obs = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{task_name}_obs.pt")    
        return obs 
    
    elif task_name in ["my_ten_twomoons", "mog_10"]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        obs = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/{task_name}_obs.pt")    
        return obs 
        
    elif task_name in ["mog_2_nabc", "mog_5_nabc", "mog_10_nabc"]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if task_name == "mog_2_nabc":
            obs = torch.load(f"{current_dir}/../depot_hyun/NeuralABC_R/MoG_2/MoG_2_x0.pt")
        elif task_name == "mog_5_nabc":
            obs = torch.load(f"{current_dir}/../depot_hyun/NeuralABC_R/MoG_5/MoG_x0.pt")           
        elif task_name == "mog_10_nabc":
            obs = torch.load(f"{current_dir}/../depot_hyun/NeuralABC_R/MoG_10/MoG_10_x0.pt")
        return torch.tensor(obs, dtype = torch.float32)

    elif task_name in ["double_slcp_summary_transform2"]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        obs = torch.load(f"{current_dir}/../depot_hyun/hyun/NPE_ABC/seeds/double_slcp_summary_transform2_obs.pt")    
        return obs

    elif task_name in ["slcp_summary_transform2", "slcp"]:
        obs_list = []
        for j in range(1, 11):
            task = sbibm.get_task("slcp")
            observation = task.get_observation(num_observation=j)  # 10 per task
            obs_list.append(observation[0].tolist())
        if task_name == "slcp_summary_transform2":
            return SLCP_summary_transform2(torch.tensor(obs_list))
        else:
            return torch.tensor(obs_list).to(torch.float32)
        
    elif task_name in ["slcp_distractors"]:
        obs_list = []
        for j in range(1, 11):
            task = sbibm.get_task("slcp_distractors")
            observation = task.get_observation(num_observation=j)  # 10 per task
            obs_list.append(observation[0].tolist())
        return torch.tensor(obs_list).to(torch.float32)
    else:
        raise ValueError(f"Unknown task name for observation_lists: {task_name}")


def simulator_bernoulli(thetas, batch_size=100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_matrix = torch.load("/home/hyun18/NPE_ABC/utils/files/design_matrix.pt", weights_only=True).to(device)

    N = thetas.size(0)
    output = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = thetas[start:end].to(device)

        psi = torch.matmul(theta_batch, design_matrix.T)
        z = torch.sigmoid(psi)
        y = (torch.rand_like(z) < z).float()

        output_batch = torch.matmul(y, design_matrix).to("cpu")
        output.append(output_batch)
        del theta_batch, psi, z, y, output_batch
        torch.cuda.empty_cache()  # Optional: free memory aggressively

    return torch.cat(output, dim=0)

def simulator_slcp_distractors(thetas, batch_size=100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thetas = thetas.to(device)

    N = thetas.size(0)
    output = []
    path = "/home/hyun18/NPE_ABC/utils/files"
    permutation_idx = torch.load(f"{path}/permutation_idx.torch", weights_only = False)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = thetas[start:end].to(device)

        # Simulate SLCP summary statistics
        slcp_stats = simulator_slcp3(theta_batch)

        # Add distractor noise
        noise = _sample_distractor(N=slcp_stats.size(0), batch_size=1_000, device=device)
        slcp_stats = torch.cat([slcp_stats, noise], dim=1)

        del theta_batch, noise
        output.append(slcp_stats[:, permutation_idx].cpu())  # Permute and move to CPU
        torch.cuda.empty_cache()  # Optional: free memory aggressively

    return torch.cat(output, dim=0)


def _sample_distractor(N: int, batch_size: int = 1000, device="cpu") -> torch.Tensor:
    """
    Sample in batches to avoid OOM for large N.
    Returns (N, 92) on CPU.
    """
    results = []
    remaining = N
    path = "/home/hyun18/NPE_ABC/utils/files/gmm.torch"
    gmm = torch.load(path, weights_only = False)

    # Extract parameters from the loaded gmm
    base = gmm.component_distribution.base_dist  # MultivariateStudentT

    df    = base.df.to(device)          # (20,)
    loc   = base.loc.to(device)         # (20, 92)
    scale = base.scale_tril.to(device)  # (20, 92, 92) — or covariance_matrix

    # Rebuild on GPU
    component_dist = dist.Independent(
        dist.MultivariateStudentT(df=df, loc=loc, scale_tril=scale),
        reinterpreted_batch_ndims=0
    )

    mixture_logits = gmm.mixture_distribution.logits.to(device)  # (20,)
    categorical = torch.distributions.Categorical(logits=mixture_logits)

    gmm_gpu = dist.MixtureSameFamily(categorical, component_dist)

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch = gmm_gpu.sample((current_batch,))   # (current_batch, 92)
        results.append(batch.cpu().to(torch.float32))
        remaining -= current_batch

    return torch.cat(results, dim=0)   # (N, 92)

def simulator_MoG(thetas, batch_size=1_000_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = thetas.size(0)
    output = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        theta_batch = thetas[start:end].to(device)

        # MoG parameters
        scale = torch.tensor([1.0, 0.1], device=device)

        # Bernoulli mask
        idx = D.Bernoulli(torch.tensor(0.5, device=device)).sample(theta_batch.shape)
        idx2 = 1.0 - idx

        # Sample from two Gaussians
        tmp1 = D.Normal(theta_batch, scale[0]).sample()
        tmp2 = D.Normal(theta_batch, scale[1]).sample()

        # Mixture
        mixed = tmp1 * idx + tmp2 * idx2

        output.append(mixed.cpu())

        # Free memory
        del theta_batch, idx, idx2, tmp1, tmp2, mixed
        torch.cuda.empty_cache()

    return torch.cat(output, dim=0)

def simulator_Lapl_5(theta: torch.Tensor, batch_size: int = 1_000_000):
    """
    Draw one Laplace sample per element of `theta`.
    
    Parameters
    ----------
    theta : (N, 5) tensor
        Location parameter of the Laplace distribution.
    batch_size : int, optional
        Max rows to process at once to control memory (default 1e6).

    Returns
    -------
    Tensor of shape (N, 5) on CPU.
    """
    if theta.ndim != 2 or theta.size(1) != 5:
        raise ValueError("theta must have shape (N, 5)")

    # Decide where to run
    device = theta.device if theta.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    theta = theta.to(device)

    # Fixed scale vector, broadcastable to (N, 5)
    b = torch.tensor([0.05, 0.10, 0.25, 0.50, 1.00],
                     dtype=theta.dtype, device=device)

    out_chunks = []
    for start in range(0, theta.size(0), batch_size):
        end = min(start + batch_size, theta.size(0))
        loc = theta[start:end]                # already on device

        dist = D.Laplace(loc=loc, scale=b)    # broadcasts automatically
        out_chunks.append(dist.sample().cpu())  # move back to CPU

        # Help Python’s GC; no need for empty_cache()
        del loc, dist

    return torch.cat(out_chunks, dim=0)

def simulator_Lapl_10(theta: torch.Tensor, batch_size: int = 1_000_000):
    """
    Draw one Laplace sample per element of `theta`.
    
    Parameters
    ----------
    theta : (N, 5) tensor
        Location parameter of the Laplace distribution.
    batch_size : int, optional
        Max rows to process at once to control memory (default 1e6).

    Returns
    -------
    Tensor of shape (N, 5) on CPU.
    """
    if theta.ndim != 2 or theta.size(1) != 10:
        raise ValueError("theta must have shape (N, 10)")

    # Decide where to run
    device = theta.device if theta.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    theta = theta.to(device)

    # Fixed scale vector, broadcastable to (N, 5)
    b = torch.tensor([0.05, 0.10, 0.25, 0.50, 1.00, 0.05, 0.10, 0.25, 0.50, 1.00],
                     dtype=theta.dtype, device=device)

    out_chunks = []
    for start in range(0, theta.size(0), batch_size):
        end = min(start + batch_size, theta.size(0))
        loc = theta[start:end]                # already on device

        dist = D.Laplace(loc=loc, scale=b)    # broadcasts automatically
        out_chunks.append(dist.sample().cpu())  # move back to CPU

        # Help Python’s GC; no need for empty_cache()
        del loc, dist

    return torch.cat(out_chunks, dim=0)

def simulator_my_twomoons(theta):
    # Local parameters specific to this simulator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = theta.to(device)
    
    a_l = -np.pi/2
    a_u = np.pi/2
    r_mu = .1
    r_sig = .01        
    
    n = theta.shape[0]
    # Use GPU tensors for distribution parameters
    a_dist = D.Uniform(torch.tensor(a_l, device=device), torch.tensor(a_u, device=device))
    r_dist = D.Normal(torch.tensor(r_mu, device=device), torch.tensor(r_sig, device=device))

    # Sample all at once on GPU
    a = a_dist.sample((n,))
    r = r_dist.sample((n,))

    # Compute px and py
    px = r * torch.cos(a) + 0.25
    py = r * torch.sin(a)

    # Compute final x and y
    x = px - torch.abs(theta.sum(dim=1)) / np.sqrt(2)
    y = py + (theta[:, 1] - theta[:, 0]) / np.sqrt(2)

    return torch.stack([x, y], dim=1).to("cpu")

def simulator_my_five_twomoons(theta):
    # theta: N * 10 dimensions
    X = []
    for i in range(5):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    return torch.cat(X, dim = 1)

def simulator_my_ten_twomoons(theta):
    # theta: N * 20 dimensions
    X = []
    for i in range(10):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    return torch.cat(X, dim = 1)

def simulator_my_fifty_twomoons(theta):
    # theta: N * 100 dimensions
    X = []
    for i in range(50):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    return torch.cat(X, dim = 1)


def simulator_my_five_twomoons_err2(theta):
    # theta: N * 10 dimensions
    X = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(5):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    batch_size  = theta.size(0)
    tmp = torch.randn( (batch_size,2), device = device) * 2.0 
    X.append(tmp.cpu())
    return torch.cat(X, dim = 1)

def simulator_my_five_twomoons_err5(theta):
    # theta: N * 10 dimensions
    X = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(5):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    batch_size  = theta.size(0)
    tmp = torch.randn( (batch_size,5), device = device) * 2.0 
    X.append(tmp.cpu())
    return torch.cat(X, dim = 1)
    
def simulator_bernoulli_glm2_err90(theta):
    # theta: N * 10 dimensions
    X = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    permute = torch.load(f"{os.path.dirname(os.path.abspath(__file__))}/../depot_hyun/hyun/NPE_ABC/seeds/bernoulli_glm2_err90_permutation.pt", weights_only = False)
    X.append(simulator_bernoulli(theta))
    batch_size  = theta.size(0)
    for _ in range(9):
        tmp = torch.randn( (batch_size,10), device = device) * 2.0 
        X.append(tmp.cpu())
    return torch.cat(X, dim = 1).cpu()[:,permute]

def simulator_bernoulli_glm2_err(theta, err_num = 90):
    # theta: N * 10 dimensions
    X = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    permute = torch.load(f"{os.path.dirname(os.path.abspath(__file__))}/../depot_hyun/hyun/NPE_ABC/seeds/bernoulli_glm2_err{err_num}_permutation.pt", weights_only = False)
    X.append(simulator_bernoulli(theta))
    batch_size  = theta.size(0)
    for _ in range(err_num // 10):
        tmp = torch.randn( (batch_size,10), device = device) * 2.0 
        X.append(tmp.cpu())
    return torch.cat(X, dim = 1).cpu()[:,permute]


def simulator_my_five_twomoons_err(theta, err_num =90):
    # theta: N * 10 dimensions
    X = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    permute = torch.load(f"{os.path.dirname(os.path.abspath(__file__))}/../depot_hyun/hyun/NPE_ABC/seeds/my_five_twomoons_err{err_num}_permutation.pt", weights_only = False)
    for i in range(5):
        tmp = torch.clone(theta[:, 2*i : (2*i + 2 )] )
        tmp2 = simulator_my_twomoons(tmp)
        X.append(tmp2)
    batch_size  = theta.size(0)
    for _ in range(err_num // 10):
        tmp = torch.randn( (batch_size,10), device = device) * 2.0 
        X.append(tmp.cpu())
    return torch.cat(X, dim = 1).cpu()[:,permute]


def simulator_slcp3(theta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    theta = theta.to(device)

    n = theta.shape[0]

    mu0 = theta[:, 0].unsqueeze(1)
    mu1 = theta[:, 1].unsqueeze(1)
    sigma0 = theta[:, 2].unsqueeze(1)
    sigma1 = theta[:, 3].unsqueeze(1)
    r = torch.tanh(theta[:, 4]).unsqueeze(1)

    # Repeat for 4 blocks
    eps0 = torch.randn(n, 4, device=theta.device)
    eps1 = torch.randn(n, 4, device=theta.device)

    # Broadcast params
    mu0 = mu0.repeat(1, 4)
    mu1 = mu1.repeat(1, 4)
    sigma0 = sigma0.repeat(1, 4)
    sigma1 = sigma1.repeat(1, 4)
    r = r.repeat(1, 4)

    x0 = mu0 + sigma0**2 * eps0
    x1 = mu1 + sigma1**2 * (r * eps0 + torch.sqrt(1 - r ** 2) * eps1)

    out = torch.stack([x0, x1], dim=2).reshape(n, -1)
    return out.cpu()

def simulator_double_slcp_summary(theta):
    # theta: N * 10 dimensions
    X = []
    for i in range(2):
        tmp = torch.clone(theta[:, 5*i : (5*i + 5 )] )
        tmp2 = simulator_slcp3(tmp)
        tmp2 = SLCP_summary_transform2(tmp2)
        X.append(tmp2)
    return torch.cat(X, dim = 1)



# ---- Channel builders ----
def channel_binary(p: float) -> torch.Tensor:
    """
    One-bit randomized response channel C(p):
      with prob p -> report truth; with prob (1-p) -> uniform random bit.
    Returns a 2x2 matrix with rows=reported, cols=true.
    """
    same = 0.5 * (1 + p)
    flip = 0.5 * (1 - p)
    C = torch.tensor([[same, flip],
                      [flip, same]], dtype=torch.double)
    return C

def channel_2x2(p1: float, p2: float) -> torch.Tensor:
    """
    Joint channel for two independent binary variables.
    If both use the same p, pass p1=p2=p.
    Returns a 4x4 matrix R with rows=reported cells, cols=true cells,
    under the fixed cell order [00, 01, 10, 11].
    """
    C1 = channel_binary(p1)  # for variable 1
    C2 = channel_binary(p2)  # for variable 2
    R = torch.kron(C1, C2)   # Kronecker product
    return R  # shape (4,4)

# ---- Simulator ----
def simulator_rr_cont_table_22(
    theta: torch.Tensor,
    p: float = 0.5,
    n: int = 400,
    batch_size: int = 1_000_000,
) -> torch.Tensor:
    """
    Simulate privatized counts y ~ Multinomial(n, q) with q = R(p) @ theta
    for a 2x2 contingency table under randomized response.

    Args:
        theta: (N, 4) tensor; each row sums to 1; order [00, 01, 10, 11].
        p: truthful-report probability for each variable (use p in [0,1]).
           If you want different ps per variable, change call to channel_2x2(p1,p2).
        n: total count per table draw (e.g., 400).
        batch_size: process rows of theta in chunks for memory efficiency.

    Returns:
        reported_counts: (N, 4) tensor of privatized counts.
    """
    assert theta.dim() == 2 and theta.size(1) == 4, "theta must be (N,4)."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = theta.to(device)
    theta = theta.to(torch.double)

    # Build joint channel R (4x4)
    R = channel_2x2(p, p).to(theta.device)  # use (p,p); change if p1!=p2

    N = theta.size(0)
    out = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        th = theta[start:end].to(device)  # (B,4)

        # Compute q = R @ th^T  -> but batched as th @ R^T  (shape (B,4))
        q = (th @ R.T).clamp(min=0)  # numerical safety
        q = q / q.sum(dim=1, keepdim=True)  # re-normalize exactly

        # Batched multinomial sampling: one draw per row
        m = Multinomial(total_count=n, probs=q)
        y = m.sample()  # (B,4)
        out.append(y.cpu())
    return torch.cat(out, dim=0).to(torch.float32)


def Simulators(task_name: str):
    task_name = task_name.lower()
    if task_name in ["bernoulli_glm2"]:
        return simulator_bernoulli
    elif task_name in ["bernoulli_glm2_err10", "bernoulli_glm2_err30", "bernoulli_glm2_err50", "bernoulli_glm2_err70", "bernoulli_glm2_err90"]:
        def simulator_with_err(theta):
            return simulator_bernoulli_glm2_err(theta, err_num = int(task_name.split("_")[-1][3:5]))
        return simulator_with_err
    elif task_name in ["my_five_twomoons_err10", "my_five_twomoons_err30", "my_five_twomoons_err50", "my_five_twomoons_err70", "my_five_twomoons_err90"]:
        def simulator_with_err(theta):
            return simulator_my_five_twomoons_err(theta, err_num = int(task_name.split("_")[-1][3:5]))
        return simulator_with_err
    elif task_name in ["two_moons"]:
        return simulator_my_twomoons
    elif task_name in ["my_twomoons"]:
        return simulator_my_twomoons
    elif task_name in ["my_five_twomoons"]:
        return simulator_my_five_twomoons
    
    elif task_name in ["table_dp_22"]:
        def cont_table_dp_generator(theta):
            return simulator_rr_cont_table_22(theta, p = 0.8, n = 4526, batch_size = 100_000)
        return cont_table_dp_generator
    
    elif task_name in ["my_ten_twomoons"]:
        return simulator_my_ten_twomoons
    elif task_name in ["my_ten_twomoons"]:
        return simulator_my_ten_twomoons
    elif task_name in ["my_fifty_twomoons"]:
        return simulator_my_fifty_twomoons
    elif task_name in ["mog_10", "mog_2_nabc", "mog_5_nabc", "mog_10_nabc"]:
        return simulator_MoG
    elif task_name in ["slcp_distractors"]:
        return simulator_slcp_distractors
    elif task_name in ["slcp"]:
        return simulator_slcp3

    elif task_name in ["slcp_summary_transform2"]:
        def summary_generator(theta):
            x = simulator_slcp3(theta)  # [N, 8]
            return SLCP_summary_transform2(x)  # [N, 5]
        return summary_generator
    elif task_name in ["double_slcp_summary_transform2"]:
        return simulator_double_slcp_summary
    else:
        raise ValueError(f"Unknown task name for simulator: {task_name}")
    
def MoG_posterior(obs, n_samples, bounds = None):
    obs = torch.tensor(obs)
    if obs.ndim == 1:
        obs = torch.reshape(obs, (1, obs.size(0)))
    scale = [1.0, 0.1]
    n_samples2 = n_samples * 1000

    idx =  D.Bernoulli(torch.tensor(1/2)).sample((n_samples2,obs.size(1) )) 
    idx2 = 1 - idx

    tmp1 = D.Normal(obs[0], torch.tensor(scale[0])).sample((n_samples2,))
    tmp2 = D.Normal(obs[0], torch.tensor(scale[1])).sample((n_samples2,))

    tmp = tmp1 * idx + tmp2 * idx2
    if bounds is not None:
        tmp = torch.clone(apply_bounds(tmp, bounds))
    sam_ind = np.random.choice(np.arange(0, tmp.size()[0]), n_samples, replace = True)
    return tmp[sam_ind,:]

def apply_bounds(samples, bounds):
    # Apply bounds to filter the samples
    if bounds is not None:
        index = []
        for j in range(samples.size()[1]):  # Iterate over each dimension
            ind = (samples[:, j] < bounds[j][1]) & (samples[:, j] > bounds[j][0])
            index.append(ind)
        index = torch.stack(index, 1)
        index = torch.all(index, 1)  # Check if all conditions hold per sample
        samples = samples[index]
    return samples

def my_twomoons_posterior(obs = torch.tensor([0.0,0.0]), n_samples = 100):
    c = 1/np.sqrt(2)
    theta = torch.zeros((n_samples, 2))
    for i in range(n_samples):
        p = Simulators("my_twomoons")(torch.zeros(1,2))
        q = torch.zeros(2)
        q[0] = p[0,0] - obs[0]
        q[1] = obs[1] - p[0,1]

        if np.random.rand() < 0.5:
            q[0] = -q[0]

        theta[i, 0] = c * (q[0] - q[1])
        theta[i, 1] = c * (q[0] + q[1])
    return theta