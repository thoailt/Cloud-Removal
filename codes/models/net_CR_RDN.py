# codes/models/net_CR_RDN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint # Only if you use gradient checkpointing
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .submodules import df_conv, df_resnet_block, kernel2d_conv

def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    # Using view and permute for pixel shuffle
    # Equivalent to nn.PixelShuffle, but custom if you need it.
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    
    # Correct order for pixel shuffle: (B, C, H_out, r, W_out, r) -> (B, C*r*r, H_out, W_out)
    # The original implementation seems to put channels*r*r in the wrong place based on typical pixel_shuffle.
    # If the original one works for you, keep it. Otherwise, consider:
    # shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, channels * upscale_factor * upscale_factor, out_height, out_width)
    # The provided function converts (B, C, H, W) to (B, C*r^2, H/r, W/r).
    # This is an INVERSE pixel shuffle or sub-pixel convolution output.
    # Let's assume the original intent for this function name is to rearrange into a smaller feature map with more channels,
    # which is often used before upsampling to then feed into a PixelShuffle layer.
    # If it's used for upsampling, it implies the output channels are C_out = C_in / (r*r).
    # The name `pixel_reshuffle` is ambiguous here. If it's used for downsampling features to be later upsampled by nn.PixelShuffle, then the name is misleading.
    # If this is indeed what is fed into `UPNet = nn.PixelShuffle(2)` later, then `input` should be `(B, C*r^2, H/r, W/r)` and output `(B, C, H, W)`.
    # Based on UPNet, `nn.PixelShuffle(2)` expects `input_channels` for `nn.Conv2d` to be `256`, and after shuffle `64`.
    # So `pixel_reshuffle` here seems to be a custom operation that increases channels and reduces spatial resolution.
    # Let's stick with the original logic for now, assuming it matches the model's design.

    channels_out = channels * upscale_factor * upscale_factor
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels_out, out_height, out_width)


class DFG(nn.Module):
    def __init__(self, channels, ks_2d):
        super(DFG, self).__init__()
        ks = 3
        half_channels = channels // 2
        self.fac_warp = nn.Sequential(
            df_conv(channels, half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_resnet_block(half_channels, kernel_size=ks),
            df_conv(half_channels, half_channels * ks_2d ** 2, kernel_size=1)
        )

    def forward(self, opt_f, sar_f):
        concat = torch.cat([opt_f, sar_f], 1)
        out = self.fac_warp(concat)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww - Using indexing='ij' for newer PyTorch versions
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_SAR = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_SAR = nn.Dropout(attn_drop)

        # Assuming output of 1x1 conv should be 1 channel (for gate), not 8
        # If num_heads is 8, and this is meant to fuse attention maps, it should be 1.
        # Let's assume it should be num_heads for now, but this might need adjustment based on paper.
        self.attn_fuse_1x1conv = nn.Conv2d(num_heads, num_heads, kernel_size=1) 


        self.proj = nn.Linear(dim, dim)
        self.proj_SAR = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_SAR = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x, x_SAR = inputs # Unpack inputs

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_SAR = self.qkv_SAR(x_SAR).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        q_SAR, k_SAR, v_SAR = qkv_SAR[0], qkv_SAR[1], qkv_SAR[2] # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        q_SAR = q_SAR * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn_SAR = (q_SAR @ k_SAR.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn_SAR = attn_SAR + relative_position_bias.unsqueeze(0)

        # Fusion logic (moved common part outside if/else for clarity)
        # Reshape attention maps to (B_*nW, nH, N, N) for the 1x1 conv
        attn_diff = attn_SAR - attn
        
        # Original: attn_diff_conv = self.attn_fuse_1x1conv(attn_SAR - attn)
        # This means attn_SAR - attn should have 8 channels.
        # However, attn and attn_SAR shapes are (B_, nH, N, N).
        # We need to reshape for 1x1 conv. (B_*nH, N, N) -> (B_*nH, 1, N, N) for conv input (if we want to fuse per head)
        # Or, (B_, nH, N, N) -> (B_, nH, N*N) -> (B_, nH, N, N) 
        # A 1x1 conv on attention map with shape (B, num_heads, N, N) should operate on the num_heads dimension.
        # This implies it should be applied to the reshaped tensor: (B, num_heads, N*N).
        # Or, the conv takes (B_, num_heads, N, N) as input directly, if nn.Conv2d allows that,
        # but typically conv is (B, C, H, W). Here H,W would be N,N and C=num_heads. This seems right.
        
        attn_diff_conv_input = attn_diff.view(B_, self.num_heads, N, N) # Make sure it's (B, C, H, W)
        attn_fuse_gate = torch.sigmoid(self.attn_fuse_1x1conv(attn_diff_conv_input))

        # Apply gate
        attn = attn + (attn_SAR - attn) * attn_fuse_gate.expand_as(attn) # expand gate to match attn shape

        if mask is not None:
            nW = mask.shape[0]
            # Apply mask AFTER fusion
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Apply softmax and dropout AFTER fusion and masking
        attn = self.softmax(attn)
        attn_SAR = self.softmax(attn_SAR) # Keep separate softmax if SAR output is needed too
                                          # but the fusion updates `attn`, so `attn_SAR` here is not the 'gated' one
                                          # if you intend to use the gated `attn` for SAR path too, you might do:
                                          # attn_SAR = attn # if they become the same after fusion
                                          # For now, stick to original logic: two separate softmaxes.

        attn = self.attn_drop(attn)
        attn_SAR = self.attn_drop_SAR(attn_SAR)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_SAR = (attn_SAR @ v_SAR).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x_SAR = self.proj_SAR(x_SAR)

        x = self.proj_drop(x)
        x_SAR = self.proj_drop_SAR(x_SAR)
        
        return [x, x_SAR]

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize, input_resolution, num_heads, window_size, shift_size,
                 mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate

        act_layer=nn.GELU

        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])
        self.conv_SAR = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

        self.dim = growRate
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_SAR = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(self.dim)
        self.norm2_SAR = norm_layer(self.dim)

        mlp_hidden_dim = int(self.dim * mlp_ratio)

        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_SAR = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask calculation (common for both paths)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, inputs):

        input_opt, input_sar = inputs # Renamed `input` to `input_opt` for clarity
        x, x_SAR = inputs # `x` is the current feature from previous RDB_Conv or SFENet2

        H, W = self.input_resolution

        x_conv = self.conv(x)
        x_SAR_conv = self.conv_SAR(x_SAR)

        # Flatten and transpose for attention (B, C, H, W) -> (B, H*W, C)
        x_conv_unfold = x_conv.flatten(2).transpose(1, 2) 
        x_SAR_conv_unfold = x_SAR_conv.flatten(2).transpose(1, 2)

        shortcut = x_conv_unfold 
        shortcut_SAR = x_SAR_conv_unfold

        B, H_W, growRate = x_conv_unfold.shape

        # Reshape to (B, H, W, C) for window partitioning
        x = x_conv_unfold.view(B, H, W, growRate) 
        x_SAR = x_SAR_conv_unfold.view(B, H, W, growRate)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_SAR = torch.roll(x_SAR, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_x_SAR = x_SAR

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, growRate) # (num_windows*B, N, C)

        x_SAR_windows = window_partition(shifted_x_SAR, self.window_size)
        x_SAR_windows = x_SAR_windows.view(-1, self.window_size * self.window_size, growRate) # (num_windows*B, N, C)

        # W-MSA/SW-MSA
        attn_opt_out, attn_sar_out = self.attn([x_windows, x_SAR_windows], mask=self.attn_mask)

        # merge windows
        attn_opt_out = attn_opt_out.view(-1, self.window_size, self.window_size, growRate)
        attn_sar_out = attn_sar_out.view(-1, self.window_size, self.window_size, growRate)
        shifted_x = window_reverse(attn_opt_out, self.window_size, H, W)
        shifted_x_SAR = window_reverse(attn_sar_out, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_SAR = torch.roll(shifted_x_SAR, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            x_SAR = shifted_x_SAR

        x = x.view(B, H_W, growRate) # Reshape back to (B, H*W, C)
        x_SAR = x_SAR.view(B, H_W, growRate) # Reshape back to (B, H*W, C)

        # FFN (Feed-Forward Network)
        # Apply normalization *before* MLP if using pre-norm Transformer style
        x = shortcut + self.drop_path(x) # Residual connection before normalization, then FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_SAR = shortcut_SAR + self.drop_path_SAR(x_SAR)
        x_SAR = x_SAR + self.drop_path_SAR(self.mlp_SAR(self.norm2_SAR(x_SAR)))

        # Reshape back to (B, C, H, W) for concatenation
        x_unfold = x.transpose(1, 2).view(B, growRate, H, W)
        x_SAR_unfold = x_SAR.transpose(1, 2).view(B, growRate, H, W)

        # Return concatenated features to grow the channel dimension (DenseNet style)
        return [torch.cat((input_opt, x_unfold), 1), torch.cat((input_sar, x_SAR_unfold), 1)]


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize, input_resolution, num_heads, window_size,
                 mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers # Number of RDB_Conv layers within this RDB

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(inChannels=G0 + c * G, growRate=G, kSize=kSize, input_resolution=input_resolution,
                                  num_heads=num_heads, window_size=window_size,
                                  shift_size=0 if (c % 2 == 0) else window_size // 2, # Apply shifted window attention
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                  attn_drop=attn_drop,
                                  drop_path=drop_path[c] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion (LFF) for both Optical and SAR paths
        self.LFF = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)
        self.LFF_SAR = nn.Conv2d(G0 + C * G, G0, 3, 1, 1)

    def forward(self, inputs):
        # inputs is a list: [optical_features, sar_features]
        x_opt_in, x_sar_in = inputs
        
        # Pass through sequence of RDB_Conv layers
        x_opt_out, x_sar_out = self.convs([x_opt_in, x_sar_in]) # self.convs is a Sequential of RDB_Conv, which returns [x, x_SAR]

        # Apply Local Feature Fusion and add residual connections
        return [self.LFF(x_opt_out) + x_opt_in, self.LFF_SAR(x_sar_out) + x_sar_in]


class RDN_residual_CR(nn.Module):
    def __init__(self, input_size):
        super(RDN_residual_CR, self).__init__()
        # Ensure input_size is a multiple of 2 (due to pixel_reshuffle by 2)
        if input_size % 2 != 0:
            raise ValueError("input_size must be a multiple of 2 for pixel_reshuffle(2).")

        self.G0 = 96 # Initial feature depth
        kSize = 3 # Kernel size for convolutions

        # number of RDB blocks, conv layers, out channels
        self.D = 6 # Number of RDBs
        self.C = 5 # Number of RDB_Conv layers per RDB
        self.G = 48 # Growth rate for RDB_Conv

        num_heads = 8  
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0. # Dropout rate for Mlp
        attn_drop_rate = 0. # Dropout rate for attention weights
        drop_path_rate = 0.2 # Stochastic depth rate
        norm_layer = nn.LayerNorm # Normalization layer for transformer blocks

        # Shallow feature extraction net for Optical data (13 bands)
        # Assuming pixel_reshuffle(cloudy_data, 2) makes channels 13*4 = 52.
        # This initial conv should take 52 channels.
        self.SFENet1 = nn.Conv2d(13 * (2*2), self.G0, 5, padding=2, stride=1) # 13 bands * (upscale_factor)^2
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Shallow feature extraction net for SAR data (2 bands)
        # Assuming pixel_reshuffle(SAR, 2) makes channels 2*4 = 8.
        self.SFENet1_SAR = nn.Conv2d(2 * (2*2), self.G0, 5, padding=2, stride=1) # 2 bands * (upscale_factor)^2
        self.SFENet2_SAR = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Stochastic depth decay rule
        # Total RDB_Conv layers = self.D * self.C
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.D * self.C)]

        # Residual Dense Blocks (RDBs)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            # Pass the corresponding slice of drop_path rates to each RDB
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C, kSize=kSize,
                    input_resolution=(input_size // 2, input_size // 2), # RDBs operate on half-res features
                    num_heads=num_heads, 
                    window_size=window_size,
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i * self.C:(i + 1) * self.C], 
                    norm_layer=norm_layer)
            )

        # Dynamic Fusion (DF)
        channels = self.G0
        ks_2d = 5 # Kernel size for kernel2d_conv used with DFG
        self.kernel_conv_ksize = ks_2d

        self.DF = nn.ModuleList() # Dynamic Filter Generators
        for i in range(self.D):
            # DFG takes concat of OPT and SAR features, both of size `channels`
            self.DF.append(DFG(channels * 2, ks_2d)) # Input to DFG is 2 * G0 channels

        # Gates for fusion
        self.sar_fuse_1x1conv = nn.ModuleList() # Gate for SAR, aggregation
        for i in range(self.D):
            self.sar_fuse_1x1conv.append(nn.Conv2d(channels, channels, kernel_size=1)) # Input/Output: G0 channels

        self.opt_distribute_1x1conv = nn.ModuleList() # Gate for OPT, distribution
        for i in range(self.D):
            self.opt_distribute_1x1conv.append(nn.Conv2d(channels, channels, kernel_size=1)) # Input/Output: G0 channels

        # Global Feature Fusion (GFF)
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling Net (UPNet)
        # Input to UPNet is G0 channels (96)
        # Conv2d(G0, 256) -> PixelShuffle(2) means 256 / (2*2) = 64 output channels
        # Then Conv2d(64, 13)
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2), # Upscales spatial resolution by 2, reduces channels by 4 (256 -> 64)
            nn.Conv2d(64, 13, kSize, padding=(kSize - 1) // 2, stride=1) # Output 13 channels
        ])

    def forward(self, cloudy_data, SAR):
        # Initial feature extraction using pixel_reshuffle (custom version)
        # This pixel_reshuffle as implemented reduces spatial size by `upscale_factor` and increases channels by `upscale_factor^2`.
        # So cloudy_data (B, 13, H, W) -> B_shuffle (B, 13*4, H/2, W/2) if upscale_factor=2
        # SAR (B, 2, H, W) -> B_shuffle_SAR (B, 2*4, H/2, W/2) if upscale_factor=2
        
        # Note: Your pixel_reshuffle is named differently and seems to act like a strided conv/downsampling op.
        # If your intention is standard pixel shuffle (upsampling), then `nn.PixelShuffle` should be used directly
        # and the input to SFENet1/SFENet1_SAR should be the original data or different processing.
        # Assuming the current pixel_reshuffle is used as intended for initial feature processing.

        initial_upscale_factor = 2 # Based on the UPNet's PixelShuffle factor and channel calculations
                                   # This initial 'pixel_reshuffle' actually performs a downsample-like operation
                                   # that increases channels. This is unconventional naming for `pixel_reshuffle`.
                                   # If it's effectively a `nn.Conv2d(in_channels, in_channels * factor**2, stride=factor)` followed by a reshape.
                                   # Let's verify input channels to SFENet1: 13 * (2*2) = 52.
                                   # So, `pixel_reshuffle` is essentially doing feature extraction and downsampling here.
        B_reshuffled_opt = pixel_reshuffle(cloudy_data, initial_upscale_factor) 
        f__1 = self.SFENet1(B_reshuffled_opt) # Output G0 channels, H/2, W/2
        x = self.SFENet2(f__1) # Output G0 channels, H/2, W/2

        B_reshuffled_sar = pixel_reshuffle(SAR, initial_upscale_factor)
        f__1__SAR = self.SFENet1_SAR(B_reshuffled_sar) # Output G0 channels, H/2, W/2
        x_SAR = self.SFENet2_SAR(f__1__SAR) # Output G0 channels, H/2, W/2

        RDBs_out = []
        for i in range(self.D):
            # Pass current optical and SAR features through RDB
            x, x_SAR = self.RDBs[i]([x, x_SAR])
            
            # Apply cross-modality fusion
            x, x_SAR = self.fuse(x, x_SAR, i)

            RDBs_out.append(x) # Collect optical features for GFF
        
        # Global Feature Fusion (GFF)
        x_gff = self.GFF(torch.cat(RDBs_out, 1)) # Concatenate all RDB optical outputs
        x_gff += f__1 # Add residual connection from SFENet1 output

        # Up-sampling Net (UPNet)
        # This UPNet includes a PixelShuffle layer for final upsampling
        pred_CloudFree_data = self.UPNet(x_gff) 
        
        # Final residual connection with original cloudy_data
        # This implies that the network learns the residual (cloudy_data - clean_data)
        # Or, if cloudy_data is full resolution and pred_CloudFree_data is also full res, this is fine.
        # But `UPNet` output is (B, 13, H, W) and original cloudy_data is (B, 13, H, W).
        # This makes sense for residual learning.
        pred_CloudFree_data = pred_CloudFree_data + cloudy_data 
        
        return pred_CloudFree_data

    def fuse(self, OPT, SAR, i):
        """
        Performs cross-modality fusion between Optical (OPT) and SAR features.
        """
        OPT_m = OPT
        SAR_m = SAR

        # Dynamic Filter Generation and Application
        # kernel_sar is generated from concatenation of OPT_m and SAR_m
        kernel_sar = self.DF[i](OPT_m, SAR_m)
        SAR_m_filtered = kernel2d_conv(SAR_m, kernel_sar, self.kernel_conv_ksize)

        # SAR Fusion Gate (from SAR_m_filtered and OPT_m to update OPT)
        sar_s = self.sar_fuse_1x1conv[i](SAR_m_filtered - OPT_m) # Calculate difference
        sar_fuse_gate = torch.sigmoid(sar_s) # Compute gate (value between 0 and 1)

        # Update OPT features using the SAR-derived gate
        new_OPT = OPT + (SAR_m_filtered - OPT_m) * sar_fuse_gate # Residual addition based on gated difference

        # OPT Distribution Gate (from new_OPT and SAR_m_filtered to update SAR)
        new_OPT_m = new_OPT # Use the newly updated OPT features
        opt_s = self.opt_distribute_1x1conv[i](new_OPT_m - SAR_m_filtered) # Calculate difference
        opt_distribute_gate = torch.sigmoid(opt_s) # Compute gate

        # Update SAR features using the OPT-derived gate
        new_SAR = SAR + (new_OPT_m - SAR_m_filtered) * opt_distribute_gate # Residual addition based on gated difference

        return new_OPT, new_SAR


# Example usage for testing the network (kept for self-contained testing)
if __name__ == "__main__":
    import argparse
    import os
    # For local testing, ensure you have basic opts attributes
    class TestOpts:
        def __init__(self, crop_size, is_use_cloudmask=True):
            self.crop_size = crop_size
            self.is_use_cloudmask = is_use_cloudmask # To avoid error if ModelCRNet checks this

    # Use a dummy opts for local test
    opts = TestOpts(crop_size=128)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # Create the model
    model = RDN_residual_CR(opts.crop_size).cuda() # Ensure model is on GPU for testing

    # Dummy input data
    # Assuming cloudy_data has 13 channels and SAR has 2 channels
    cloudy = torch.rand(1, 13, 128, 128).cuda()
    # cloudfree = torch.rand(1, 13, 128, 128).cuda() # Not needed for forward pass
    s1_sar = torch.rand(1, 2, 128, 128).cuda()

    # Perform a forward pass
    pred_cloudfree = model(cloudy, s1_sar)

    print(f"Input cloudy shape: {cloudy.shape}")
    print(f"Input SAR shape: {s1_sar.shape}")
    print(f"Predicted cloud-free shape: {pred_cloudfree.shape}")

    # Verify output size matches input size (if no explicit upsampling in RDN)
    # The UPNet has PixelShuffle(2), so it upsamples the (H/2, W/2) features back to (H, W).
    # Thus, the output should match input spatial dimensions.
    assert pred_cloudfree.shape == cloudy.data.shape, "Output shape does not match input shape!"
    print("Test passed: Output shape matches input shape.")