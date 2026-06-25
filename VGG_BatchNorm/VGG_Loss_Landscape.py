"""
VGG Loss Landscape Analysis for Task 2.
Compares VGG_A (without BN) vs VGG_A_BatchNorm (with BN).

Experiments:
  1. Train both models with multiple learning rates
  2. Record per-step training loss
  3. Plot loss landscape: min-max envelope to show loss stability
  4. Compare training curves between BN and non-BN models
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm

import argparse
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader


# ============================================================
# Utility Functions
# ============================================================
def get_accuracy(model, data_loader, device=None):
    """Calculate classification accuracy on a dataset."""
    if device is None:
        device = torch.device('cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def set_random_seeds(seed_value=0, device=None):
    """Set random seeds for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device is not None and str(device) != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Training with per-step loss recording
# ============================================================
def train_with_loss_recording(model, optimizer, criterion, train_loader,
                               epochs_n=20, model_name='model',
                               save_path=None):
    """
    Train model and record per-step loss values.
    Returns per-epoch loss arrays for loss landscape analysis.
    """
    model.to(device)
    epoch_losses = []   # average loss per epoch
    step_losses = []    # loss of every step

    for epoch in tqdm(range(epochs_n), unit='epoch', desc=f'Training {model_name}'):
        model.train()
        epoch_loss = 0.0
        epoch_step_losses = []
        batches_n = 0

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_step_losses.append(loss.item())
            batches_n += 1

        avg_loss = epoch_loss / max(batches_n, 1)
        epoch_losses.append(avg_loss)
        step_losses.extend(epoch_step_losses)

        # Validation
        val_acc = get_accuracy(model, val_loader, device)
        train_acc = get_accuracy(model, train_loader, device)
        print(f"  Epoch {epoch+1}/{epochs_n} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save model
    if save_path:
        torch.save(model.state_dict(), save_path)

    return epoch_losses, step_losses


# ============================================================
# Main Experiment: Loss Landscape Comparison
# ============================================================
def run_loss_landscape_experiment(model_class, model_name, lr_list,
                                   epochs=20, seed=2020):
    """
    Train the same model architecture with different learning rates.
    Returns list of per-step loss arrays (one per learning rate).
    """
    all_losses = []  # list of (step_losses) arrays, one per LR

    for lr in lr_list:
        print(f"\n{'='*60}")
        print(f"{model_name} | LR = {lr}")
        print(f"{'='*60}")

        set_random_seeds(seed_value=seed, device=str(device))
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        save_path = os.path.join(models_path, f'{model_name}_lr{lr}_e{epochs}.pth')
        epoch_losses, step_losses = train_with_loss_recording(
            model, optimizer, criterion, train_loader,
            epochs_n=epochs, model_name=f'{model_name}_lr{lr}',
            save_path=save_path
        )
        all_losses.append(step_losses)

    return all_losses


def compute_envelope(all_losses):
    """
    Given a list of per-step loss arrays (from different LRs),
    compute the min and max curves by trimming to the shortest length.
    """
    min_len = min(len(l) for l in all_losses)
    trimmed = [l[:min_len] for l in all_losses]
    stacked = np.array(trimmed)  # shape: (n_models, n_steps)
    max_curve = stacked.max(axis=0)
    min_curve = stacked.min(axis=0)
    return min_curve, max_curve


# ============================================================
# Plotting
# ============================================================
def plot_loss_landscape_comparison(losses_no_bn, losses_bn, lr_list,
                                    save_path=None):
    """
    Plot loss landscape comparison: VGG_A vs VGG_A_BatchNorm.
    Shows the min-max envelope for each model.
    """
    min_no_bn, max_no_bn = compute_envelope(losses_no_bn)
    min_bn, max_bn = compute_envelope(losses_bn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: VGG_A (no BN)
    ax = axes[0]
    x = np.arange(len(min_no_bn))
    ax.fill_between(x, min_no_bn, max_no_bn, alpha=0.3, color='red')
    ax.plot(x, min_no_bn, 'r-', linewidth=0.5, alpha=0.5)
    ax.plot(x, max_no_bn, 'r-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'VGG-A (without BN)\nLRs: {lr_list}')
    ax.grid(True, alpha=0.3)

    # Right: VGG_A_BatchNorm (with BN)
    ax = axes[1]
    x = np.arange(len(min_bn))
    ax.fill_between(x, min_bn, max_bn, alpha=0.3, color='blue')
    ax.plot(x, min_bn, 'b-', linewidth=0.5, alpha=0.5)
    ax.plot(x, max_bn, 'b-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'VGG-A-BN (with BN)\nLRs: {lr_list}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss landscape saved to {save_path}")
    plt.close()


def plot_combined_envelope(losses_no_bn, losses_bn, lr_list, save_path=None):
    """
    Combined plot: overlay BN vs non-BN envelopes on the same axes.
    """
    min_no_bn, max_no_bn = compute_envelope(losses_no_bn)
    min_bn, max_bn = compute_envelope(losses_bn)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Non-BN
    x_a = np.arange(len(min_no_bn))
    ax.fill_between(x_a, min_no_bn, max_no_bn, alpha=0.2, color='red',
                    label='VGG-A (no BN)')
    ax.plot(x_a, np.array([l[:len(min_no_bn)] for l in losses_no_bn]).mean(axis=0),
            'r-', linewidth=2, label='VGG-A mean')

    # BN
    x_b = np.arange(len(min_bn))
    ax.fill_between(x_b, min_bn, max_bn, alpha=0.2, color='blue',
                    label='VGG-A-BN (with BN)')
    ax.plot(x_b, np.array([l[:len(min_bn)] for l in losses_bn]).mean(axis=0),
            'b-', linewidth=2, label='VGG-A-BN mean')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Landscape: VGG-A vs VGG-A-BatchNorm\n'
                 f'(LRs: {lr_list}, {len(lr_list)} runs each)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined envelope saved to {save_path}")
    plt.close()


def plot_epoch_loss_comparison(epoch_losses_no_bn, epoch_losses_bn,
                                lr_list, save_path=None):
    """
    Plot per-epoch average loss curves for each LR side by side.
    """
    n_lrs = len(lr_list)
    fig, axes = plt.subplots(1, n_lrs, figsize=(5*n_lrs, 4))
    if n_lrs == 1:
        axes = [axes]

    for i, lr in enumerate(lr_list):
        ax = axes[i]
        epochs = range(1, len(epoch_losses_no_bn[i]) + 1)
        ax.plot(epochs, epoch_losses_no_bn[i], 'r-o', markersize=4,
                linewidth=1.5, label='VGG-A (no BN)')
        ax.plot(epochs, epoch_losses_bn[i], 'b-s', markersize=4,
                linewidth=1.5, label='VGG-A-BN (with BN)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'LR = {lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training Loss Comparison: VGG-A vs VGG-A-BN', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Epoch loss comparison saved to {save_path}")
    plt.close()


# ============================================================
# Run Experiment
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 2: BN Loss Landscape')
    parser.add_argument('--small', action='store_true',
                        help='Use 15000 training samples + 10 epochs for quick test')
    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    home_path = os.path.dirname(os.path.abspath(__file__))
    figures_path = os.path.join(home_path, 'figures')
    models_path = os.path.join(home_path, 'saved_models')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Data — small mode uses subset for speed
    batch_size = 128
    num_workers = 0
    if args.small:
        n_items_train = 15000
        epochs = 10
        n_items_val = 2000
    else:
        n_items_train = -1
        epochs = 20
        n_items_val = -1

    train_loader = get_cifar_loader(root='./data/', train=True, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, n_items=n_items_train)
    val_loader = get_cifar_loader(root='./data/', train=False, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, n_items=n_items_val)

    # Learning rates to test (different step sizes)
    lr_list = [1e-3, 2e-3, 5e-4, 1e-4]

    print("=" * 70)
    print("TASK 2: BATCH NORMALIZATION - LOSS LANDSCAPE ANALYSIS")
    print(f"Learning rates: {lr_list}")
    print(f"Epochs per run: {epochs}")
    print(f"Training samples: {n_items_train if n_items_train > 0 else 'full (50K)'}")
    print(f"Total runs: {len(lr_list) * 2}")
    print("=" * 70)

    # ---- Experiment 1: VGG_A (without BN) ----
    print("\n>>> Phase 1: Training VGG-A (without BN) with multiple LRs")
    epoch_losses_no_bn = []
    step_losses_no_bn = []

    for lr in lr_list:
        print(f"\n--- VGG_A | LR={lr} ---")
        set_random_seeds(seed_value=2020, device=str(device))
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        save_path = os.path.join(models_path, f'VGG_A_lr{lr}_e{epochs}.pth')
        e_losses, s_losses = train_with_loss_recording(
            model, optimizer, criterion, train_loader,
            epochs_n=epochs, model_name=f'VGG_A_lr{lr}',
            save_path=save_path
        )
        epoch_losses_no_bn.append(e_losses)
        step_losses_no_bn.append(s_losses)

        # Evaluate
        acc = get_accuracy(model, val_loader, device)
        print(f"  Final Val Accuracy (VGG_A, lr={lr}): {acc:.4f}")

    # ---- Experiment 2: VGG_A_BatchNorm (with BN) ----
    print("\n>>> Phase 2: Training VGG-A-BN (with BN) with multiple LRs")
    epoch_losses_bn = []
    step_losses_bn = []

    for lr in lr_list:
        print(f"\n--- VGG_A_BatchNorm | LR={lr} ---")
        set_random_seeds(seed_value=2020, device=str(device))
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        save_path = os.path.join(models_path, f'VGG_A_BN_lr{lr}_e{epochs}.pth')
        e_losses, s_losses = train_with_loss_recording(
            model, optimizer, criterion, train_loader,
            epochs_n=epochs, model_name=f'VGG_A_BN_lr{lr}',
            save_path=save_path
        )
        epoch_losses_bn.append(e_losses)
        step_losses_bn.append(s_losses)

        # Evaluate
        acc = get_accuracy(model, val_loader, device)
        print(f"  Final Val Accuracy (VGG_A_BN, lr={lr}): {acc:.4f}")

    # ---- Phase 3: Plot Results ----
    print("\n>>> Phase 3: Generating Plots")

    # 3a: Side-by-side envelope
    plot_loss_landscape_comparison(
        step_losses_no_bn, step_losses_bn, lr_list,
        save_path=os.path.join(figures_path, 'loss_landscape_side_by_side.png')
    )

    # 3b: Combined envelope (key figure)
    plot_combined_envelope(
        step_losses_no_bn, step_losses_bn, lr_list,
        save_path=os.path.join(figures_path, 'loss_landscape_combined.png')
    )

    # 3c: Per-epoch loss curves
    plot_epoch_loss_comparison(
        epoch_losses_no_bn, epoch_losses_bn, lr_list,
        save_path=os.path.join(figures_path, 'epoch_loss_comparison.png')
    )

    # ---- Phase 4: Summary ----
    print("\n" + "=" * 70)
    print("TASK 2 EXPERIMENT COMPLETE")
    print(f"Figures saved to: {figures_path}")
    print(f"Models saved to: {models_path}")
    print("=" * 70)

    # Print summary stats
    print("\n--- Loss Envelope Width (max - min) ---")
    for i, lr in enumerate(lr_list):
        s_bn = np.array(step_losses_bn[i])
        s_no_bn = np.array(step_losses_no_bn[i])
        print(f"LR={lr}: VGG_A range={s_no_bn.max()-s_no_bn.min():.4f}, "
              f"VGG_A_BN range={s_bn.max()-s_bn.min():.4f}")

    # Compare envelope widths
    min_nb, max_nb = compute_envelope(step_losses_no_bn)
    min_b, max_b = compute_envelope(step_losses_bn)
    width_nb = (max_nb - min_nb).mean()
    width_b = (max_b - min_b).mean()
    print(f"\nAverage envelope width (VGG_A, no BN):  {width_nb:.4f}")
    print(f"Average envelope width (VGG_A_BN, w/ BN): {width_b:.4f}")
    if width_b < width_nb:
        print("=> BN reduces loss variance across learning rates ✓")
    else:
        print("=> Note: BN may need more epochs to show effect clearly")
