"""
Loss Landscape visualization: compare loss stability with/without BN.
Plots min-max envelope of loss across different learning rates.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os


def plot_loss_landscape(all_losses: List[np.ndarray],
                        labels: List[str],
                        save_path: str,
                        title: str = 'Loss Landscape'):
    """
    Plot loss landscape as min-max envelope.

    Args:
        all_losses: list of (steps,) or (epochs,) arrays, one per model/lr
        labels: legend labels for each curve
        save_path: output path
        title: plot title
    """
    if len(all_losses) != len(labels):
        raise ValueError("all_losses and labels must have same length")

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_losses)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (losses, label) in enumerate(zip(all_losses, labels)):
        x = np.arange(len(losses))
        ax.plot(x, losses, color=colors[i], linewidth=1.5, alpha=0.8,
                label=label)

    ax.set_xlabel('Training Step / Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss landscape saved to {save_path}")


def plot_loss_envelope(all_loss_curves: List[np.ndarray],
                       label: str,
                       save_path: str,
                       color: str = 'blue',
                       title: str = 'Loss Landscape Envelope'):
    """
    Plot min-max envelope for multiple runs of the SAME model.
    Shows how stable the loss is across different initializations.

    Args:
        all_loss_curves: list of (steps,) arrays from N training runs
        label: model label
        save_path: output path
        color: fill color
        title: plot title
    """
    all_loss_curves = [np.array(l) for l in all_loss_curves]
    n_steps = min(len(l) for l in all_loss_curves)

    # Truncate all to same length
    curves = np.array([l[:n_steps] for l in all_loss_curves])
    mean_curve = curves.mean(axis=0)
    min_curve = curves.min(axis=0)
    max_curve = curves.max(axis=0)
    x = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(x, min_curve, max_curve, alpha=0.3, color=color,
                    label=f'{label} (min-max range)')
    ax.plot(x, mean_curve, color=color, linewidth=2,
            label=f'{label} (mean)')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss envelope saved to {save_path}")


def plot_double_envelope(curves_a: List[np.ndarray],
                         curves_b: List[np.ndarray],
                         label_a: str, label_b: str,
                         save_path: str,
                         title: str = 'Loss Landscape Comparison'):
    """
    Compare loss envelopes of TWO models (e.g., with and without BN).
    """
    def prepare(curves):
        curves = [np.array(l) for l in curves]
        n = min(len(l) for l in curves)
        c = np.array([l[:n] for l in curves])
        return c.mean(0), c.min(0), c.max(0)

    mean_a, min_a, max_a = prepare(curves_a)
    mean_b, min_b, max_b = prepare(curves_b)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(mean_a))
    ax.fill_between(x, min_a, max_a, alpha=0.2, color='red')
    ax.plot(x, mean_a, 'r-', linewidth=2, label=label_a)

    x = np.arange(len(mean_b))
    ax.fill_between(x, min_b, max_b, alpha=0.2, color='blue')
    ax.plot(x, mean_b, 'b-', linewidth=2, label=label_b)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison envelope saved to {save_path}")
