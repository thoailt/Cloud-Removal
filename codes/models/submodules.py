# codes/submodules.py
#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import torch.nn as nn
import torch
import numpy as np # Not directly used in the final version of `kernel2d_conv`, but often handy
import torch.nn.functional as F

# Consider defining LEAKY_VALUE as a default parameter in df_conv/df_ResnetBlock if it's specific
# Or, if it's a global constant, keep it here.
LEAKY_VALUE = 0.1 # This is fine as a module-level constant

def df_conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """
    A convolution block consisting of a 2D convolution and a LeakyReLU activation.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(LEAKY_VALUE, inplace=True)
    )

class df_ResnetBlock(nn.Module):
    """
    A ResNet block with two convolution layers and a LeakyReLU activation,
    followed by a skip connection.
    """
    def __init__(self, in_channels, kernel_size, dilation=(1, 1), bias=True): # Default dilation for clarity
        super(df_ResnetBlock, self).__init__()
        # Ensure dilation is a tuple of two elements
        if not isinstance(dilation, (list, tuple)) or len(dilation) != 2:
            raise ValueError("dilation must be a tuple or list of two integers for df_ResnetBlock.")
            
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, 
                      dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(LEAKY_VALUE, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, 
                      dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x # Residual connection
        return out

# The helper function df_resnet_block is redundant but kept if there's a reason for it
def df_resnet_block(in_channels, kernel_size=3, dilation=(1,1), bias=True):
    """
    Helper function to create a df_ResnetBlock.
    (This function is redundant if df_ResnetBlock is always called directly,
    but can be useful as a factory if different types of ResnetBlocks are introduced).
    """
    return df_ResnetBlock(in_channels, kernel_size, dilation, bias=bias)


def kernel2d_conv(feat_in, kernel, ksize):
    """
    Performs 2D convolution with a dynamically generated kernel.
    This is a Python implementation, potentially slower than a custom CUDA layer.

    Args:
        feat_in (torch.Tensor): Input feature map of shape (N, C, H, W).
        kernel (torch.Tensor): Dynamic kernels of shape (N, C_out * ksize * ksize, H, W).
                                Note: C_out here is the same as C of feat_in.
        ksize (int): Size of the square kernel (e.g., 3 for 3x3).

    Returns:
        torch.Tensor: Output feature map after dynamic convolution (N, C, H, W).
    """
    channels = feat_in.size(1) # Number of input channels
    N, kernels_flat_channels, H, W = kernel.size() # kernels_flat_channels is C * ksize * ksize
    pad = (ksize - 1) // 2

    # 1. Pad input feature map
    feat_in_padded = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    
    # 2. Unfold input features into patches
    # feat_in_unfolded: (N, C, H_out, W_out, ksize, ksize)
    feat_in_unfolded = feat_in_padded.unfold(2, ksize, 1).unfold(3, ksize, 1)
    
    # Rearrange and reshape to (N, H, W, C, ksize*ksize)
    feat_in_reshaped = feat_in_unfolded.permute(0, 2, 3, 1, 5, 4).contiguous() # (N, H, W, C, ksize, ksize)
    feat_in_reshaped = feat_in_reshaped.view(N, H, W, channels, -1) # (N, H, W, C, ksize*ksize)

    # 3. Reshape dynamic kernels
    # kernel: (N, C * ksize * ksize, H, W) -> (N, H, W, C * ksize * ksize)
    kernel_reshaped = kernel.permute(0, 2, 3, 1).contiguous()
    # Now reshape to (N, H, W, C, ksize*ksize)
    kernel_reshaped = kernel_reshaped.view(N, H, W, channels, ksize, ksize)
    kernel_reshaped = kernel_reshaped.permute(0, 1, 2, 3, 5, 4).contiguous().view(N, H, W, channels, -1)
    
    # 4. Perform element-wise multiplication and sum over the kernel dimension
    feat_out = torch.sum(feat_in_reshaped * kernel_reshaped, -1)
    
    # 5. Rearrange back to (N, C, H, W)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out