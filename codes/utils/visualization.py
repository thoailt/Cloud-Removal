# codes/utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image
import torch # Để xử lý torch.Tensor

def convert_to_rgb(data):
    """
    Convert image data (torch.Tensor or np.ndarray) to a numpy RGB array (H, W, 3)
    with values in the range [0, 255] and dtype np.uint8.
    Special handling for Sentinel-2 13-band data and general multi-channel/grayscale images.
    
    Args:
        data (torch.Tensor or np.ndarray): Image data with shape (B, C, H, W) or (C, H, W).
                                           Assumes pixel values are in [0, 1] (float).

    Returns:
        np.ndarray: RGB image with shape (H, W, 3), values in [0, 255], dtype np.uint8.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if data.ndim == 4:  # (B, C, H, W)
        data = data[0]  

    # Handle 13-band data (Sentinel-2) or general multi-channel data
    if data.shape[0] == 13: # 13-band Sentinel-2 data
        # Assume bands 4, 3, 2 correspond to R, G, B (0-indexed: 3, 2, 1)
        red = data[3, :, :]
        green = data[2, :, :]
        blue = data[1, :, :]
    # elif data.shape[0] == 2: # Color RGB for 2 channels (e.g., SAR VV/VH)
    #     red = data[0, :, :]
    #     green = data[1, :, :]
    #     blue = np.zeros_like(data[0, :, :]) 
    elif data.shape[0] == 2:
        grayscale_channel = np.mean(data, axis=0)
        red = grayscale_channel
        green = grayscale_channel
        blue = grayscale_channel
    elif data.shape[0] >= 3:
        red = data[0, :, :]
        green = data[1, :, :]
        blue = data[2, :, :]
    elif data.shape[0] == 1:
        red = green = blue = data[0, :, :]
    else:
        raise ValueError(f"Unsupported number of channels: {data.shape[0]}. "
                 "Expected 1, 2, 3, or 13 channels.")  # Updated error message

    rgb = np.stack([red, green, blue], axis=-1)
    
    rgb_min = np.min(rgb)
    rgb_max = np.max(rgb)
    if rgb_max - rgb_min > 1e-8:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb = np.zeros_like(rgb)

    rgb = (rgb * 255).astype(np.uint8)
    return rgb

def create_image_grid_with_titles(cloud_data_rgb, sar_data_rgb, predicted_rgb, ground_truth_rgb, cloud_mask_np):
    """
    Create an image grid with titles and return as a PIL image for logging to Wandb.
    
    Args:
        cloud_data_rgb (np.ndarray): Cloudy optical image (RGB, uint8).
        sar_data_rgb (np.ndarray): SAR image (RGB or grayscale, uint8).
        predicted_rgb (np.ndarray): Predicted cloud-free image (RGB, uint8).
        ground_truth_rgb (np.ndarray): Ground truth cloud-free image (RGB, uint8).
        cloud_mask_np (np.ndarray): Cloud mask (H, W) with values 0 (clear), 1 (cloud), -1 (shadow).

    Returns:
        PIL.Image: Combined grid image with titles.
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) # Increase size for better readability
    titles = [
        "Cloudy (RGB)",
        "SAR (RGB)",
        "Cloud Mask (Overlay)",
        "Predicted (RGB)",
        "Ground Truth (RGB)",
        "Cloud Mask (Raw)"
    ]
    
    cloud_data_norm = cloud_data_rgb.astype(np.float32) / 255.0
    
    overlay_red = np.zeros_like(cloud_mask_np, dtype=np.float32)
    overlay_green = np.zeros_like(cloud_mask_np, dtype=np.float32)
    overlay_blue = np.zeros_like(cloud_mask_np, dtype=np.float32)

    # Get boolean masks for cloud and shadow
    is_cloud = (cloud_mask_np == 1)
    is_shadow = (cloud_mask_np == -1)

    # Assign color to each channel using 2D boolean masks
    # Cloud in red: [1, 0, 0]
    overlay_red[is_cloud] = 1.0
    overlay_green[is_cloud] = 0.0
    overlay_blue[is_cloud] = 0.0

    # Shadow in blue: [0, 0, 1]
    overlay_red[is_shadow] = 0.0
    overlay_green[is_shadow] = 0.0
    overlay_blue[is_shadow] = 1.0

    # Stack channels to create 3D color overlay
    mask_overlay_color_channels = np.stack([overlay_red, overlay_green, overlay_blue], axis=-1)

    # Combine original image and mask overlay
    # Ensure cloud_data_norm is float32 and in [0,1]
    combined_overlay_image = (0.7 * cloud_data_norm + 0.3 * mask_overlay_color_channels)
    combined_overlay_image = np.clip(combined_overlay_image, 0, 1)
    combined_overlay_image = (combined_overlay_image * 255).astype(np.uint8)

    # Create visualization for raw cloud mask for "Cloud Mask (Raw)" image
    cloud_mask_colored = np.zeros((*cloud_mask_np.shape, 3), dtype=np.uint8)
    cloud_mask_colored[cloud_mask_np == 0] = [0, 0, 0]       # Clear: Black
    cloud_mask_colored[cloud_mask_np == 1] = [255, 255, 255] # Cloud: White
    cloud_mask_colored[cloud_mask_np == -1] = [0, 0, 255]    # Shadow: Blue

    images_to_display = [
        cloud_data_rgb,
        sar_data_rgb,
        combined_overlay_image,
        predicted_rgb,
        ground_truth_rgb,
        cloud_mask_colored
    ]

    for i, (image, title) in enumerate(zip(images_to_display, titles)):
        ax = axs[i // 3, i % 3]
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0] # Remove batch dimension if present
        ax.imshow(image)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PIL.Image.open(buf)