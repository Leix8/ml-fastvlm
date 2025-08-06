import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor):
    assert tensor1.shape == tensor2.shape, f"tensor1.shape={tensor1.shape}, tensor2.shape={tensor2.shape}, Tensors must have the same shape"
    
    # Cosine similarity across channel dimension
    cos_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)  # → [H, W]

    # Euclidean distance across channels
    l2_dist = torch.norm(tensor1 - tensor2, dim=0)  # → [H, W]

    return cos_sim.cpu().numpy(), l2_dist.cpu().numpy()

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

def visualize_and_save(cos_sim, l2_dist, output_path="feature_diff_combined.png"):
    """
    Accepts PyTorch tensors or NumPy arrays and saves side-by-side heatmaps.
    """

    # Convert tensors to numpy if needed
    if isinstance(cos_sim, torch.Tensor):
        cos_sim = cos_sim.detach().cpu().numpy()
    if isinstance(l2_dist, torch.Tensor):
        l2_dist = l2_dist.detach().cpu().numpy()

    # Remove batch dimension if present: (1, H, W) -> (H, W)
    if cos_sim.ndim == 3 and cos_sim.shape[0] == 1:
        cos_sim = cos_sim.squeeze(0)
    if l2_dist.ndim == 3 and l2_dist.shape[0] == 1:
        l2_dist = l2_dist.squeeze(0)

    # If still 1D or 3D, reduce to 2D by mean or reshape
    if cos_sim.ndim == 1:
        cos_sim = cos_sim.reshape(1, -1)
    elif cos_sim.ndim == 3:
        cos_sim = cos_sim.mean(axis=-1)  # or axis=0 depending on the context

    if l2_dist.ndim == 1:
        l2_dist = l2_dist.reshape(1, -1)
    elif l2_dist.ndim == 3:
        l2_dist = l2_dist.mean(axis=-1)  # or axis=0 depending on the context

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(cos_sim, cmap='viridis', vmin=-1, vmax=1)
    axs[0].set_title("Cosine Similarity")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(l2_dist, cmap='hot')
    axs[1].set_title("L2 Distance")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved combined visualization to: {output_path}")