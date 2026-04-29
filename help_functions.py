import torch
import numpy as np
import math
import time

def ABC_rej(x0, X_cal, tol, device):
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2, 1))

    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    
    # Select points within tolerance and return to CPU if needed
    return wt1.cpu()


def compute_mad(X):
    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        X = X.to('cuda')

    # Compute the median for each column
    medians = torch.median(X, dim=0).values  # Shape: (num_columns,)

    # Compute the absolute deviations from the median
    abs_deviation = torch.abs(X - medians)  # Broadcasting over rows

    # Compute the MAD for each column
    mad = torch.median(abs_deviation, dim=0).values  # Shape: (num_columns,)
    torch.cuda.empty_cache()
    
    # Return the result on the CPU
    return mad.cpu()


def ABC_rej2(x0, X_cal, tol, device, dist_output = None):
    # Move all tensors to the target device at once
    x0 = x0.to(device)
    X_cal = X_cal.to(device)
    mad = compute_mad(X_cal)
    mad = torch.reshape(mad, (1, X_cal.size(1))).to(device)
    dist = torch.sqrt(torch.mean(torch.abs(X_cal.to(device) - x0.to(device))**2/mad**2, 1))
    
    # Determine threshold distance using top-k rather than sorting the entire tensor
    num = X_cal.size(0)
    nacc = int(num * tol)
    ds = torch.topk(dist, nacc, largest=False).values[-1]
    
    # Create mask and filter based on the threshold distance
    wt1 = (dist <= ds)
    torch.cuda.empty_cache()
    # Select points within tolerance and return to CPU if needed
    if dist_output is not None:
        return wt1.cpu(), ds, mad.cpu()
    else:
        return wt1.cpu()


def TABC_Jacobian(s, theta, sobs, density_estimator, device = "cpu"):
    density_estimator = density_estimator.to(device).eval()
    flow = density_estimator.net
    transform = flow._transform
    embed = flow._embedding_net

    
    if theta.ndim == 1:
        theta  = torch.reshape(theta, (1, theta.size(0)))
    if s.ndim == 1:
        s = torch.reshape(s, (1, s.size(0)))
    if sobs.ndim == 1:
        sobs = torch.reshape(sobs, (1, sobs.size(0)))
    
    with torch.no_grad():
        
        z, _ = transform.forward(theta.to(device), context = embed(s.to(device)))

        _, numerator = transform.inverse(z, context = embed(sobs.to(device)))
        _, denominator = transform.inverse(z, context = embed(s.to(device)))

    return numerator/denominator


def fisher_z(x, eps=1e-6):
    x = torch.clamp(x, -1+eps, 1-eps)        # or: x = x*(1-eps)
    z = 0.5 * torch.log((1 + x) / (1 - x))   # = atanh(x)
    return z

def log1p(x):
    return torch.log(1+x)

def log1p2(x):
    return torch.log(.1+x)


def SLCP_summary_transform2(X):
    """
    Compute summary statistics for SLCP data:
    - Means and standard deviations for even and odd indexed dimensions
    - Average correlation between even and odd groups

    Args:
        X: Tensor of shape [N, 8]

    Returns:
        Tensor of shape [N, 5] containing m0, m1, s0, s1, and rho
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    X0 = X[:, [0, 2, 4, 6]]
    X1 = X[:, [1, 3, 5, 7]]
    m0 = X0.mean(dim=1, keepdim=True)
    m1 = X1.mean(dim=1, keepdim=True)
    s0 = X0.std(dim=1, correction=0, keepdim=True)
    s1 = X1.std(dim=1, correction=0, keepdim=True)
    
    # Compute correlation per sample
    cov = ((X0 - m0) * (X1 - m1)).mean(dim=1, keepdim=True)
    rho = cov / (s0 * s1 + 1e-12)
    rho = torch.clamp(rho, -1.0, 1.0)

    s0 = log1p2(s0)
    s1 = log1p2(s1)
    rho = fisher_z(rho)

    return torch.cat((m0, m1, s0, s1, rho), dim=1).cpu()





class UnifSample:
    def __init__(self, bins = 10):
        self.bins = bins
        
    def box(self, sample ,num):
        heights, intervals = np.histogram(sample, self.bins, density = True)
        a, b = UnifSample.support_index(heights)
        samples = UnifSample.support_sample(intervals, a, b, num)
        return torch.tensor(samples, dtype = torch.float32)
    
    @staticmethod
    def support_index(heights):
        temp = ( (heights /np.sum(heights) ) != 0.0)
        return intervals_connect(heights,temp)
        
    @staticmethod
    def support_sample(intervals, a, b, num):
        interval_diffs = intervals[b] - intervals[a]
        prop = interval_diffs / np.sum(interval_diffs)
        size_num = np.random.multinomial(num, prop)
        
        # Preallocate an array for the results
        ran = np.empty(num)
        
        cum_sum = 0
        for i, size in enumerate(size_num):
            if size > 0:
                # Generate samples in the current interval
                tmp = np.random.uniform(0, 1, size) * interval_diffs[i] + intervals[a][i]
                ran[cum_sum:cum_sum + size] = tmp
                cum_sum += size
        
        np.random.shuffle(ran)
        return ran[:num]

def param_box(unifsam, sample, num):
    """
    unifsam: UnifSample object with determined seeds
    sample : n*p tensor
    num : the number of samples
    """
    theta_new = []
    for j in range(sample.size()[1]):
        sam = sample[:,j]
        theta_new.append(torch.reshape(unifsam.box(sam, num), (num, 1)))
        del sam
    return torch.cat(theta_new, 1)


def intervals_connect(heights, indices):
    a = list()
    b = list()
    for i in range(len(heights)):
        if indices[i] == True:
            if i == 0:
                a.append(i)
            elif indices[i-1] == False:
                a.append(i)
            if i == len(heights)-1:
                b.append(i+1)
            elif indices[i+1] == False:
                b.append(i+1)
    return [a,b]


def truncated_normal(shape, mean=0.0, std=1.0, lower=-0.5, upper=0.5):
    """
    Generates samples from a truncated normal distribution in O(1) time using inverse CDF method.

    Returns:
    - Tensor of shape `shape` with samples from the truncated normal distribution.
    """
    # Convert lower and upper bounds to standard normal space
    lower_cdf = 0.5 * (1 + math.erf((lower - mean) / (std * math.sqrt(2))))
    upper_cdf = 0.5 * (1 + math.erf((upper - mean) / (std * math.sqrt(2))))

    # Sample uniformly in the truncated CDF range
    uniform_samples = torch.rand(shape, dtype=torch.float32) * (upper_cdf - lower_cdf) + lower_cdf

    # Apply inverse CDF (probit function) using erfinv
    truncated_samples = mean + std * torch.erfinv(2 * uniform_samples - 1) * math.sqrt(2)

    return truncated_samples

def truncated_mvn_sample(L, mean, std, lower, upper):
    """
    L: size of priors
    mean, std, lower, upper: torch.tensor with size [d]
    """
    d = mean.size(0)
    samples = []
    for j in range(d):
        tmp = truncated_normal((L,), mean[j], std[j], lower[j], upper[j])
        samples.append(tmp)
    return torch.column_stack(samples)


def truncated_mvn_log_prob(x, mean, std, lower, upper):
    """
    Computes log probability of x under a truncated multivariate normal distribution
    with independent dimensions.
    
    Parameters:
    - x:     Tensor of shape (N, d) or (d,)
    - mean, std, lower, upper: Tensor of shape (d,)
    
    Returns:
    - log_prob: Tensor of shape (N,) or scalar
    """
    # Log normalizing constant per dimension: log(Phi(upper) - Phi(lower))
    lower_cdf = 0.5 * (1 + torch.erf((lower - mean) / (std * math.sqrt(2))))
    upper_cdf = 0.5 * (1 + torch.erf((upper - mean) / (std * math.sqrt(2))))
    log_norm = torch.log(upper_cdf - lower_cdf)  # shape (d,)

    # Log normal pdf per dimension
    log_pdf = -0.5 * ((x - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)  # shape (N, d)

    # Mask out-of-support values
    in_support = (x >= lower) & (x <= upper)  # shape (N, d)
    log_prob_per_dim = torch.where(in_support, log_pdf - log_norm, torch.tensor(float('-inf')))  # shape (N, d)

    return log_prob_per_dim.sum(dim=-1)  # shape (N,)

@torch.no_grad()
def inverse_from_Z_chunked(
    density_estimator,
    x_b,                    # [B, x_dim]
    theta_dim,
    N,
    chunk_elems=131072,     # rows of (N*B) per chunk
    verbose=True,           # turn on/off prints
    logger=None,            # optional: a logging.Logger
    log_every=10,            # print every k chunks
):
    def log(msg):
        if not verbose and logger is None:
            return
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    density_estimator.eval()
    flow = density_estimator.net
    flow.eval()
    
    
    transform = flow._transform
    embed = flow._embedding_net
    
    device = next(flow.parameters()).device
    x_b = x_b.to(device)
    
    with torch.no_grad():    
        context = embed(x_b)                     # [B, context_dim]
    B = context.shape[0]
    
    ctx_flat = context.unsqueeze(0).expand(N, B, -1).reshape(N * B, context.shape[-1])
    #ctx_flat = torch.repeat_interleave(context,repeats = N, dim = 0)
    total = N * B
    n_chunks = math.ceil(total / chunk_elems)

    theta_flat = torch.empty(total, theta_dim, device=device, dtype=x_b.dtype)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated(device)
    t0 = time.perf_counter()

    log(f"[start] device={device} | total_rows={total} (= N*B = {N}*{B}) | "
        f"theta_dim={theta_dim} | context_dim={context.shape[-1]} | "
        f"chunk_elems={chunk_elems} | n_chunks={n_chunks}")

    
    processed = 0
    for ci in range(n_chunks):
        start = ci * chunk_elems
        end   = min(start + chunk_elems, total)
        rows  = end - start

        t_chunk0 = time.perf_counter()
        ctx_chunk = ctx_flat[start:end].contiguous()    # [rows, context_dim]
        z_chunk = torch.randn_like(ctx_chunk)
    
        with torch.no_grad():
            y_chunk, _ = transform.inverse(z_chunk, context=ctx_chunk)

        theta_flat[start:end] = y_chunk

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_chunk1 = time.perf_counter()

        processed += rows
        if (ci % log_every) == 0:
            if device.type == "cuda":
                cur_mem = torch.cuda.memory_allocated(device)
                max_mem = torch.cuda.max_memory_allocated(device)
                mem_str = f"mem(cur={cur_mem/1e6:.1f}MB, max={max_mem/1e6:.1f}MB, +{(cur_mem-start_mem)/1e6:.1f}MB)"
            else:
                mem_str = "mem(cpu)"

            log(f"[chunk {ci+1}/{n_chunks}] rows={rows}, "
                f"range=[{start}:{end}) | "
                f"elapsed={t_chunk1 - t_chunk0:.3f}s | "
                f"progress={processed}/{total} ({100*processed/total:.1f}%) | {mem_str}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    log(f"[forward done] total_time={t1 - t0:.3f}s | throughput={(processed/(t1-t0))/1e6:.2f}M rows/s")

    theta = theta_flat.reshape(N, B, theta_dim)
    return theta


@torch.no_grad()
def forward_from_theta_test(
    density_estimator,
    x_b,                    # [B, x_dim]
    theta_test,             # [N, theta_dim]
    chunk_elems=1231072,
    verbose=True,
    logger=None,
    log_every=10,
):
    def log(msg):
        if not verbose and logger is None:
            return
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    density_estimator.eval()
    flow = density_estimator.net
    flow.eval()

    N, theta_dim = theta_test.shape
    device = next(flow.parameters()).device

    x_b = x_b.to(device)
    theta_test = theta_test.to(device)

    transform = flow._transform
    embed = flow._embedding_net

    context = embed(x_b)                  # [B, context_dim]
    B, ctx_dim = context.shape

    total = N * B
    n_chunks = math.ceil(total / chunk_elems)

    # Output stored chunk by chunk, never allocate [N*B, ...] upfront
    Z_out = torch.empty(N, B, theta_dim, device=device, dtype=x_b.dtype)

    log(f"[start] device={device} | N={N} | B={B} | total_rows={total} | "
        f"theta_dim={theta_dim} | ctx_dim={ctx_dim} | "
        f"chunk_elems={chunk_elems} | n_chunks={n_chunks}")

    t0 = time.perf_counter()
    processed = 0

    for ci in range(n_chunks):
        start = ci * chunk_elems
        end   = min(start + chunk_elems, total)
        rows  = end - start

        # Convert flat indices -> (n_idx, b_idx) without materializing full tensors
        n_idx = torch.arange(start, end, device=device) // B   # [rows]
        b_idx = torch.arange(start, end, device=device) % B    # [rows]

        theta_chunk = theta_test[n_idx]   # [rows, theta_dim]  — index, no expand
        ctx_chunk   = context[b_idx]      # [rows, ctx_dim]    — index, no expand

        t_chunk0 = time.perf_counter()
        with torch.no_grad():
            z_chunk, _ = transform.forward(theta_chunk, context=ctx_chunk)
        # Write directly into the right slice of Z_out
        Z_out.view(total, theta_dim)[start:end] = z_chunk
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_chunk1 = time.perf_counter()

        processed += rows
        if ci % log_every == 0:
            if device.type == "cuda":
                cur_mem  = torch.cuda.memory_allocated(device)
                max_mem  = torch.cuda.max_memory_allocated(device)
                mem_str  = f"mem(cur={cur_mem/1e6:.1f}MB, max={max_mem/1e6:.1f}MB)"
            else:
                mem_str = "mem(cpu)"
            log(f"[chunk {ci+1}/{n_chunks}] rows={rows} [{start}:{end}) | "
                f"elapsed={t_chunk1-t_chunk0:.3f}s | "
                f"progress={processed}/{total} ({100*processed/total:.1f}%) | {mem_str}")

    t1 = time.perf_counter()
    log(f"[done] total_time={t1-t0:.3f}s | "
        f"throughput={(processed/(t1-t0))/1e6:.2f}M rows/s")

    return Z_out  # [N, B, theta_dim]


def covs_chunked(MAT, chunk=1000):
    N = MAT.shape[0]
    mu = MAT.mean(dim=0, keepdim=True)         # (1, M, 2)
    covs_list = []
    for s in range(0, MAT.shape[1], chunk):
        e = s + chunk
        xc = MAT[:, s:e, :] - mu[:, s:e, :]    # (N, chunk, 2)
        cov_chunk = torch.einsum('nmd,nme->mde', xc, xc) / (N - 1)  # (chunk,2,2)
        covs_list.append(cov_chunk)
    return torch.cat(covs_list, dim=0)       # (M, 2, 2)
