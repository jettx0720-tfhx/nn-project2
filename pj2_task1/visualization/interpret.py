"""
Model interpretation visualizations: confusion matrix, misclassified samples.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from utils.metrics import confusion_matrix, CIFAR10_CLASSES


def plot_confusion_matrix(test_labels, test_preds, save_path: str,
                          title: str = 'Confusion Matrix',
                          normalize: bool = True):
    """
    Plot confusion matrix as heatmap.

    Args:
        test_labels: true labels tensor
        test_preds: predicted labels tensor
        save_path: output path
        title: plot title
        normalize: if True, normalize each row to sum to 1
    """
    cm = confusion_matrix(test_preds, test_labels, num_classes=10).numpy()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    # Add text in cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_misclassified(model, test_loader, device, save_path: str,
                       n: int = 20, title: str = 'Misclassified Examples'):
    """
    Show the most confidently misclassified test images.

    Args:
        model: trained model
        test_loader: test DataLoader
        device: torch device
        save_path: output path
        n: number of examples to show
        title: plot title
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y
            pred = model(x)
            probs = torch.softmax(pred, dim=1)
            conf, preds = probs.max(1)
            preds = preds.cpu()
            conf = conf.cpu()

            for i in range(len(y)):
                if preds[i] != y[i]:
                    misclassified.append({
                        'image': x[i].cpu(),
                        'true': y[i].item(),
                        'pred': preds[i].item(),
                        'confidence': conf[i].item(),
                    })
            if len(misclassified) >= n * 2:
                break

    # Sort by confidence (most confident mistakes)
    misclassified.sort(key=lambda x: x['confidence'], reverse=True)
    misclassified = misclassified[:n]

    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        if idx < len(misclassified):
            m = misclassified[idx]
            img = m['image'] * std + mean
            img = img.permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img)
            ax.set_title(f"T:{CIFAR10_CLASSES[m['true']]}\nP:{CIFAR10_CLASSES[m['pred']]}",
                         fontsize=8, color='red')
        ax.axis('off')

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified examples saved to {save_path}")


def plot_param_vs_accuracy(results_list, save_path: str):
    """
    Scatter plot: test accuracy vs parameter count, annotated with model names.
    Key scoring visualization for the report.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results_list:
        params = int(float(r['num_params']))
        acc = float(r['test_acc'])
        name = r['model_name']
        ax.scatter(params, acc, s=100, alpha=0.7)
        ax.annotate(name, (params, acc),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, alpha=0.8)

    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Performance vs Parameter Count')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Param vs Accuracy plot saved to {save_path}")
