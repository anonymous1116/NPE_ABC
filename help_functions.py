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


def ABC_rej2(x0, X_cal, tol, device, case = None):
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
    del mad, dist
    # Select points within tolerance and return to CPU if needed
    return wt1.cpu()



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


@torch.no_grad()
def forward_from_Z_chunked(
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
