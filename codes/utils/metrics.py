# codes/utils/metrics.py
import math
import torch
import torch.nn.functional as F
# from torch.autograd import Variable # Variable is deprecated, use torch.Tensor directly
import numpy as np
# from math import exp # math.exp is for scalar, torch.exp is for tensor. gaussian uses scalar.

def gaussian(window_size, sigma):
    """
    Generates a 1D Gaussian kernel.
    """
    # Using torch.arange for tensor operations is often preferred
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window for SSIM calculation.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # Do not use Variable. Expand the tensor.
    # The .cuda() call will be removed, as the Trainer/Model will handle device placement.
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def SSIM(img1, img2, window_size=11, size_average=True):
    """
    Calculates Structural Similarity Index (SSIM) between two images.
    Assumes img1 and img2 are torch.Tensor of shape (B, C, H, W) and values are in [0, 1].
    """
    (_, channel, _, _) = img1.size()
    # Move window to the same device as images
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8) # Add epsilon to avoid division by zero

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def MAE(img1, img2):
    """
    Calculates Mean Absolute Error (MAE).
    Args:
        img1 (torch.Tensor): Predicted tensor.
        img2 (torch.Tensor): Ground truth tensor.
    Returns:
        float: MAE value.
    """
    mae = torch.mean(torch.abs(img1 - img2))
    return mae.item() # Return scalar float value

def MSE(img1, img2):
    """
    Calculates Mean Squared Error (MSE).
    Args:
        img1 (torch.Tensor): Predicted tensor.
        img2 (torch.Tensor): Ground truth tensor.
    Returns:
        float: MSE value.
    """
    mse = torch.mean((img1 - img2) ** 2)
    return mse.item() # Return scalar float value

def PSNR(img1, img2, PIXEL_MAX=1.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    Args:
        img1 (torch.Tensor): Predicted tensor.
        img2 (torch.Tensor): Ground truth tensor.
        PIXEL_MAX (float): Maximum possible pixel value. Default is 1.0 for [0, 1] normalized data.
    Returns:
        float: PSNR value. Returns 100 if MSE is 0.
    """
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100.0 # Return float for consistency
    
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse.item())) # Use mse.item() for math.sqrt
    return psnr # Return only PSNR scalar value