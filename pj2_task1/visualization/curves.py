"""
Training curve visualization: loss and accuracy over epochs.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def plot_training_curves(history: Dict, save_path: str,
                         title: str = None):
    """
    Plot train/val loss and accuracy curves for a single run.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_comparison_curves(histories: Dict[str, Dict], save_path: str,
                           metric: str = 'val_acc',
                           title: str = 'Comparison'):
    """
    Overlay a metric from multiple runs for comparison.

    Args:
        histories: {name: history_dict} mapping
        metrics: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: output path
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], '-', color=color, label=name,
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison curves saved to {save_path}")


def plot_phase_summary(results: List[Dict], x_col: str, y_col: str,
                       labels: List[str], save_path: str,
                       title: str = '', xlabel: str = '', ylabel: str = ''):
    """Bar chart comparing results across experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(results))
    values = [float(r[y_col]) for r in results]
    bars = ax.bar(x, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(results))))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel or y_col)
    ax.set_xlabel(xlabel or x_col)
    ax.set_title(title)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Phase summary saved to {save_path}")
