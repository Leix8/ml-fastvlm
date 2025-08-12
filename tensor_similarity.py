import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import matplotlib.ticker as mticker


def find_closest_square_factors(n):
    # Start from sqrt(n) and go down to find integer factor
    sqrt_n = int(math.sqrt(n))
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i, n // i  # H, W
    return 1, n  # fallback

def compute_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor, dim = 1): 
    assert tensor1.shape == tensor2.shape, f"tensor1.shape={tensor1.shape}, tensor2.shape={tensor2.shape}, Tensors must have the same shape"
    
    # Cosine similarity across channel dimension
    cos_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=dim)  # → [H, ]

    # Euclidean distance across channels
    # normalize
    # tensor1_normed = torch.nn.functional.normalize(tensor1, dim=1)
    # tensor2_normed = torch.nn.functional.normalize(tensor2, dim=1)
    # l2_dist = torch.norm(tensor1_normed - tensor2_normed, dim=1)

    #l2 norm, cannot eval diff relavant to value range, using normed diff instead
    l2_dist = torch.norm(tensor1 - tensor2, dim=dim)  # → [H, ]
    l2_dist[l2_dist < 1e-6] = 0.0  # Optional cleanup for small value instability

    #Gives a percentage difference in magnitude. 0 → same magnitude, higher = bigger scale mismatch.
    scale_diff = (tensor1.norm(dim=dim) - tensor2.norm(dim=dim)).abs() / ((tensor1.norm(dim=dim) + tensor2.norm(dim=dim)) / 2)

    H, W = find_closest_square_factors(cos_sim.shape[0])
    cos_sim = cos_sim.view(H, W)
    l2_dist = l2_dist.view(H, W)
    scale_diff = scale_diff.view(H, W)

    return cos_sim.cpu().numpy(), l2_dist.cpu().numpy(), scale_diff.cpu().numpy()

def reshape_to_image_grid(tensor_2d: np.ndarray, image_aspect_ratio: tuple[int, int]):
    """
    Reshape a 2D tensor (e.g. [tokens, dim]) into an image-like grid 
    according to a specified aspect ratio (W, H), without changing values.

    Args:
        tensor_2d (np.ndarray): 2D tensor of shape [N, D]
        image_aspect_ratio (tuple): (W, H) original image aspect ratio

    Returns:
        np.ndarray: reshaped 2D grid for visualization
    """
    if len(tensor_2d.shape) != 2:
        raise ValueError("Expected a 2D tensor input")

    N, D = tensor_2d.shape
    target_ratio = image_aspect_ratio[0] / image_aspect_ratio[1]

    # Try to reshape based on D as width reference
    h = int(np.sqrt(N * D / target_ratio))
    w = int(h * target_ratio)

    flat = tensor_2d.flatten()

    padded_len = w * h
    if flat.size < padded_len:
        # pad with zeros to fit the shape
        flat = np.pad(flat, (0, padded_len - flat.size))
    else:
        # truncate
        flat = flat[:padded_len]

    grid = flat.reshape(h, w)
    return grid

def visualize_and_save(cos_sim = None, l2_dist = None, scale_diff = None, image_path=None, output_path="feature_diff_combined.png", mode = None):
    """
    Visualize similarity and distance heatmaps, optionally alongside the original image.

    Parameters:
    - cos_sim: Cosine similarity tensor or array (H, W)
    - l2_dist: L2 distance tensor or array (H, W)
    - image_path: Optional path to image (for visualization)
    - output_path: Path to save the combined plot
    """

    fig_cnt = sum(x is not None for x in (image_path, cos_sim, l2_dist, scale_diff))
    fig_cnt = max(fig_cnt, 2) #supress error if only plot for one axs
    fig, axs = plt.subplots(1, fig_cnt, figsize=(6 * fig_cnt, 6))
    
    counter = 0        
    # Prepare figure layout
    if image_path is not None:
        # Load and resize image
        img = Image.open(image_path).convert("RGB")
        # img_resized = img.resize((cos_sim.shape[1], cos_sim.shape[0]))
        img_array = np.array(img)

        axs[counter].imshow(img_array)
        axs[counter].set_title("Original Image")
        axs[counter].axis("off")
        counter += 1

    if cos_sim is not None:
        # Convert tensors to numpy if needed
        if isinstance(cos_sim, torch.Tensor):
            cos_sim = cos_sim.detach().cpu().numpy()
        if cos_sim.ndim == 3 and cos_sim.shape[0] == 1:
            cos_sim = cos_sim.squeeze(0)
        # If still 1D or 3D, reduce to 2D
        if cos_sim.ndim == 1:
            cos_sim = cos_sim.reshape(1, -1)
        elif cos_sim.ndim == 3:
            cos_sim = cos_sim.mean(axis=-1)

        im1 = axs[counter].imshow(cos_sim, cmap='viridis', vmin=-1, vmax=1)
        axs[counter].set_title("Cosine Similarity")
        plt.colorbar(im1, ax=axs[counter], fraction=0.046, pad=0.04)
        counter += 1

    if l2_dist is not None:
        if isinstance(l2_dist, torch.Tensor):
            l2_dist = l2_dist.detach().cpu().numpy()
        if l2_dist.ndim == 3 and l2_dist.shape[0] == 1:
            l2_dist = l2_dist.squeeze(0)
        if l2_dist.ndim == 1:
            l2_dist = l2_dist.reshape(1, -1)
        elif l2_dist.ndim == 3:
            l2_dist = l2_dist.mean(axis=-1)

        im2 = axs[counter].imshow(l2_dist, cmap='hot')
        axs[counter].set_title("l2 distance")
        plt.colorbar(im2, ax=axs[counter], fraction=0.046, pad=0.04)
        counter += 1

    if scale_diff is not None:
        if isinstance(scale_diff, torch.Tensor):
            scale_diff = scale_diff.detach().cpu().numpy()
        # Remove batch dimension if present: (1, H, W) -> (H, W)
        if scale_diff.ndim == 3 and scale_diff.shape[0] == 1:
            scale_diff = scale_diff.squeeze(0)
        if scale_diff.ndim == 1:
            scale_diff = scale_diff.reshape(1, -1)
        elif scale_diff.ndim == 3:
            scale_diff = scale_diff.mean(axis=-1)

        im3 = axs[counter].imshow(scale_diff, cmap='hot')
        axs[counter].set_title("Magnitude Diff %")
        cbar = plt.colorbar(im3, ax=axs[counter], fraction=0.046, pad=0.04)
        # Format the labels as percentages
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        # Optional: label
        cbar.set_label("Magnitude Diff (%)")

    plt.suptitle(mode, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved combined visualization to: {output_path}")