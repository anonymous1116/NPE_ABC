import torch
import os, sys
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Dirichlet, Multinomial
from sbi.utils import BoxUniform
import sbibm
# Optional: you can use this from torch.distributions if available
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


def Bounds(task_name: str):
    task_name = task_name.lower()
    if task_name == "bernoulli_glm2":
        return None
    elif task_name in ["two_moons"]:
        return [[-1, 1]] * 2
    else:
        raise ValueError(f"Unknown task name for bounds: {task_name}")

def Priors(task_name: str):
    task_name = task_name.lower()
    if task_name in ["bernoulli_glm2"]:
        dim = 10
        loc = torch.zeros(dim)
        precision_diag = 0.5 * torch.ones(dim)
        precision_matrix = torch.diag(precision_diag)
        return MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
    elif task_name in ["two_moons"]:
        return BoxUniform(low = -1*torch.ones(2), high = 1*torch.ones(2))
    else:
        raise ValueError(f"Unknown task name for prior: {task_name}")
    
class true_Posteriors:
    def __init__(self, task):
        self.task = task

    def __call__(self, obs=None, n_samples=100, bounds=None, **kwargs):
        # Handle the case where task is 'slcp' differently
        if self.task == "two_moons":
            return self.two_moons(kwargs.get('j', 0))

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

    def my_twomoons(self, obs = torch.tensor([0.0,0.0]), n_samples = 100):
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
    
    def two_moons(self, j):
        task = sbibm.get_task("two_moons")
        return task.get_reference_posterior_samples(num_observation=j)

    def gaussian_mixture(self, j):
        task = sbibm.get_task("gaussian_mixture")  # See sbibm.get_available_tasks() for all tasks
        return task.get_reference_posterior_samples(num_observation=j)
    
    def gaussian_linear_uniform(self, j):
        task = sbibm.get_task("gaussian_linear_uniform")  # See sbibm.get_available_tasks() for all tasks
        return task.get_reference_posterior_samples(num_observation=j)
    
    def bernoulli_glm(self, j):
        try:
            # Get the directory of the current file (simulator.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, f"../depot_hyun/NeuralABC_R/bernoulli_glm/post_{j}.pt")
            post_sample = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File for posterior not found.")
        return post_sample
    
def observation_lists(task_name:str):
    task_name = task_name.lower()
    if task_name in ["two_moons", "bernoulli_glm2"]:
        obs_list = []
        for j in range(1, 11):
            task = sbibm.get_task(task_name)
            observation = task.get_observation(num_observation=j)  # 10 per task
            obs_list.append(observation[0].tolist())
        return torch.tensor(obs_list)
        
def simulator_bernoulli(thetas, batch_size=100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_matrix = torch.load("/home/hyun18/NDP/benchmark/design_matrix.pt").to(device)

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
    
def Simulators(task_name: str):
    task_name = task_name.lower()
    if task_name in ["bernoulli_glm2"]:
        return simulator_bernoulli
    elif task_name in ["two_moons"]:
        return simulator_my_twomoons
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