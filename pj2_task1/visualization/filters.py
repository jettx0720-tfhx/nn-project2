"""
Convolutional filter visualization for CIFAR-10 models.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_conv1_filters(model, save_path: str, title: str = None,
                       max_filters: int = 64):
    """
    Visualize first-layer Conv2d filters as RGB images.

    Args:
        model: PyTorch model (nn.Module)
        save_path: output image path
        title: plot title
        max_filters: maximum number of filters to display
    """
    # Find the first Conv2d layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        print("No Conv2d layer found in model")
        return

    weights = first_conv.weight.data.cpu().numpy()  # [out_ch, in_ch, h, w]
    out_ch = min(weights.shape[0], max_filters)

    # Normalize each filter to [0, 1] for visualization
    grid_size = int(np.ceil(np.sqrt(out_ch)))
    fig, axes = plt.subplots(grid_size, grid_size,
                              figsize=(grid_size*1.5, grid_size*1.5))

    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        if i < out_ch:
            filt = weights[i]  # [in_ch, h, w]
            if filt.shape[0] == 3:
                # RGB filter: transpose to [h, w, 3] and normalize
                filt = np.transpose(filt, (1, 2, 0))
                filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                ax.imshow(filt)
            else:
                # Grayscale: show first channel
                ax.imshow(filt[0], cmap='gray')
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Filter visualization saved to {save_path}")


def plot_filter_comparison(models_dict: dict, save_path: str,
                           max_filters: int = 16):
    """
    Compare first-layer filters across multiple models.

    Args:
        models_dict: {name: model} mapping
        save_path: output path
        max_filters: max filters per model
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models, max_filters,
                              figsize=(max_filters * 0.8, n_models * 0.8))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for row, (name, model) in enumerate(models_dict.items()):
        first_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                break

        if first_conv is None:
            continue

        weights = first_conv.weight.data.cpu().numpy()
        n_show = min(weights.shape[0], max_filters)

        for col in range(max_filters):
            ax = axes[row, col]
            if col < n_show:
                filt = weights[col]
                if filt.shape[0] == 3:
                    filt = np.transpose(filt, (1, 2, 0))
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                    ax.imshow(filt)
                else:
                    ax.imshow(filt[0], cmap='gray')
            ax.axis('off')

        axes[row, 0].set_ylabel(name, fontsize=10, rotation=0,
                                 ha='right', va='center')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Filter comparison saved to {save_path}")
