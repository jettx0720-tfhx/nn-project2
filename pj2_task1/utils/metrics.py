"""
Evaluation metrics for CIFAR-10 classification.
"""

import torch


def accuracy(preds, labels):
    """Compute overall accuracy."""
    return (preds == labels).float().mean().item()


def per_class_accuracy(preds, labels, num_classes=10):
    """Compute per-class accuracy, return list of length num_classes."""
    accs = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((preds[mask] == c).float().mean().item())
    return accs


def confusion_matrix(preds, labels, num_classes=10):
    """Compute confusion matrix (num_classes x num_classes)."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t.long(), p.long()] += 1
    return cm


def per_class_precision_recall(preds, labels, num_classes=10):
    """Compute per-class precision and recall."""
    precision = []
    recall = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(p)
        recall.append(r)
    return precision, recall


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
