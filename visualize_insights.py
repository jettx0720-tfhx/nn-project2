"""
Generate insight visualizations for the best trained models.
Includes: filter visualization, confusion matrix, misclassified samples, loss landscape.

Usage:
    python visualize_insights.py
"""

import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout
from models.vgg_variants import VGG_Half, VGG_Quarter
from models.residual import VGG_Residual
from models.simple_cnn import SimpleCNN
from data.loaders import get_cifar_loader, get_train_val_loaders
from visualization.filters import plot_conv1_filters, plot_filter_comparison
from visualization.interpret import (plot_confusion_matrix, plot_misclassified,
                                      plot_param_vs_accuracy)
from visualization.landscape import plot_loss_landscape, plot_double_envelope
from visualization.curves import plot_training_curves
from experiments.results import load_results
from utils.training import evaluate
from utils.metrics import CIFAR10_CLASSES


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model_from_checkpoint(model_cls, checkpoint_path, device):
    """Load a saved model from checkpoint."""
    model = model_cls()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_filter_visualizations(device):
    """Generate filter visualizations for key models."""
    print("\n--- Filter Visualizations ---")
    os.makedirs('./figures/', exist_ok=True)

    model_configs = {
        'VGG_A': (VGG_A, './saved_models/P1_VGG_A_baseline.pth'),
        'VGG_A_BatchNorm': (VGG_A_BatchNorm, './saved_models/P4_arch_VGG_A_BatchNorm.pth'),
        'VGG_Residual': (VGG_Residual, './saved_models/P4_arch_VGG_Residual.pth'),
        'SimpleCNN': (SimpleCNN, './saved_models/P4_arch_SimpleCNN.pth'),
    }

    loaded_models = {}
    for name, (cls, path) in model_configs.items():
        if os.path.exists(path):
            model = load_model_from_checkpoint(cls, path, device)
            loaded_models[name] = model
            plot_conv1_filters(model, f'./figures/filters_{name}.png',
                              title=f'{name} First-Layer Filters')
        else:
            print(f"Checkpoint not found: {path}, initializing fresh model")
            model = cls()
            model.to(device)
            loaded_models[name] = model
            plot_conv1_filters(model, f'./figures/filters_{name}.png',
                              title=f'{name} First-Layer Filters (untrained)')

    # Comparison plot
    if len(loaded_models) >= 2:
        plot_filter_comparison(loaded_models, './figures/filters_comparison.png',
                              max_filters=16)


def generate_interpretability(device):
    """Generate confusion matrix and misclassified samples for best model."""
    print("\n--- Interpretability Visualizations ---")

    # Try loading best checkpoint (VGG_A_BatchNorm > VGG_A fallback)
    model_cls = VGG_A_BatchNorm
    checkpoint = './saved_models/P4_arch_VGG_A_BatchNorm.pth'
    if not os.path.exists(checkpoint):
        checkpoint = './saved_models/P1_VGG_A_baseline.pth'
        model_cls = VGG_A  # match model class to checkpoint
    if not os.path.exists(checkpoint):
        print("No checkpoint found, skipping interpretability")
        return

    model = load_model_from_checkpoint(model_cls, checkpoint, device)

    # Get test loader
    test_loader = get_cifar_loader(
        root='./data/', batch_size=64, train=False, shuffle=False,
        use_augmentation=False
    )

    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds,
                          './figures/confusion_matrix.png',
                          title=f'Confusion Matrix (Test Acc: {test_acc:.2%})')

    # Misclassified samples
    plot_misclassified(model, test_loader, device,
                       './figures/misclassified.png', n=20,
                       title=f'Most Confidently Misclassified')


def generate_loss_landscape(device):
    """Generate loss landscape comparison: VGG_A vs VGG_A_BatchNorm."""
    print("\n--- Loss Landscape ---")

    # Load training histories from experiment results
    results = load_results()

    # Find VGG_A and VGG_A_BatchNorm results
    vgg_a_hist = None
    vgg_bn_hist = None

    # Check CSV for these experiments
    for r in results:
        if r.get('experiment_id') == 'P1_VGG_A_baseline':
            vgg_a_hist = r
        if r.get('experiment_id') == 'P4_arch_VGG_A_BatchNorm':
            vgg_bn_hist = r

    if vgg_a_hist:
        print(f"VGG_A: Test Acc={float(vgg_a_hist['test_acc']):.4f}, "
              f"Params={int(float(vgg_a_hist['num_params'])):,}")

    if vgg_bn_hist:
        print(f"VGG_A_BatchNorm: Test Acc={float(vgg_bn_hist['test_acc']):.4f}, "
              f"Params={int(float(vgg_bn_hist['num_params'])):,}")

    # Plot param vs accuracy scatter
    arch_results = [r for r in results if r.get('phase') == 'architecture']
    if arch_results:
        plot_param_vs_accuracy(arch_results, './figures/param_vs_accuracy.png')

    # If we have training histories with per-epoch loss, plot them
    import json
    for exp_id in ['P1_VGG_A_baseline', 'P4_arch_VGG_A_BatchNorm']:
        json_path = f'./experiments/results_{exp_id}.json'
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            history = data.get('history', {})
            if 'train_loss' in history:
                plot_training_curves(history,
                                    f'./figures/curves_{exp_id}.png',
                                    title=f'Training Curves: {exp_id}')


def main():
    device = get_device()
    print(f"Device: {device}")

    generate_filter_visualizations(device)
    generate_interpretability(device)
    generate_loss_landscape(device)

    print("\nAll visualizations generated in ./figures/")


if __name__ == '__main__':
    main()
