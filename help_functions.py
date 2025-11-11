import torch


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

