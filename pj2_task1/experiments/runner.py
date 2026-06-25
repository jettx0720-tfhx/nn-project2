"""
ExperimentRunner: unified interface to run a single experiment
and record results for Task 1.
"""

import torch
import torch.nn as nn
import time
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Type

from utils.training import set_random_seeds, train_full
from utils.metrics import per_class_accuracy, CIFAR10_CLASSES


@dataclass
class ExperimentConfig:
    """Configuration for one training experiment."""
    # Experiment identity
    experiment_id: str
    phase: str  # e.g., 'baseline', 'activation', 'loss', 'architecture', 'optimizer'

    # Model
    model_name: str
    model_class: Type[nn.Module]

    # Training
    epochs: int = 30
    batch_size: int = 64

    # Optimization
    optimizer_name: str = 'Adam'
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # for SGD

    # Loss
    loss_fn: str = 'CrossEntropy'
    l1_lambda: float = 0.0
    label_smoothing: float = 0.0

    # Activation (for record-keeping; activation is baked into model)
    activation: str = 'ReLU'

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'CosineAnnealingLR'  # or 'StepLR', 'ReduceLROnPlateau'

    # Seed
    seed: int = 42

    # Output
    save_dir: str = './saved_models/'
    results_csv: str = './experiments/results.csv'


def build_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Instantiate and return model from config."""
    model = config.model_class()
    model.to(device)
    return model


def build_optimizer(model: nn.Module, config: ExperimentConfig):
    """Build optimizer from config."""
    if config.optimizer_name == 'Adam':
        return torch.optim.Adam(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_name == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_name == 'SGD':
        return torch.optim.SGD(
            model.parameters(), lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def build_criterion(config: ExperimentConfig):
    """Build loss function from config."""
    if config.loss_fn == 'CrossEntropy':
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    else:
        raise ValueError(f"Unknown loss: {config.loss_fn}")


def build_scheduler(optimizer, config: ExperimentConfig):
    """Build learning rate scheduler from config."""
    if not config.use_scheduler:
        return None
    if config.scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.epochs // 3, gamma=0.1
        )
    else:
        return None


def run_experiment(config: ExperimentConfig,
                   train_loader, val_loader, test_loader,
                   device: torch.device,
                   verbose: bool = True) -> Dict:
    """
    Run a single experiment and return results dict.

    Args:
        config: ExperimentConfig with all hyperparameters
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        verbose: print progress

    Returns:
        dict with all results and metadata
    """
    # Set seed
    set_random_seeds(config.seed, str(device))

    # Build model, optimizer, criterion, scheduler
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    criterion = build_criterion(config)
    scheduler = build_scheduler(optimizer, config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Save path
    save_path = os.path.join(config.save_dir, f"{config.experiment_id}.pth")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Experiment: {config.experiment_id}")
        print(f"Phase: {config.phase}")
        print(f"Model: {config.model_name} | Params: {num_params:,}")
        print(f"Optimizer: {config.optimizer_name} | LR: {config.lr} | WD: {config.weight_decay}")
        print(f"Loss: {config.loss_fn} | L1: {config.l1_lambda} | LS: {config.label_smoothing}")
        print(f"Activation: {config.activation} | Epochs: {config.epochs}")
        print(f"{'='*70}")

    # Train
    start_time = time.time()
    history, test_loss, test_acc, test_preds, test_labels, best_val_acc = train_full(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=config.epochs,
        save_path=save_path,
        verbose=verbose,
        l1_lambda=config.l1_lambda,
    )
    train_time = time.time() - start_time

    # Per-class accuracy
    class_accs = per_class_accuracy(test_preds, test_labels)

    # Compile results
    results = {
        **asdict(config),
        'num_params': num_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_time_seconds': train_time,
        'history': {k: v for k, v in history.items()},  # lists
        'class_accs': class_accs,
        'class_names': CIFAR10_CLASSES,
    }

    # Remove non-serializable fields
    results.pop('model_class', None)

    if verbose:
        print(f"\nResults: Best Val={best_val_acc:.4f}, Test={test_acc:.4f}")
        print(f"Training time: {train_time:.1f}s ({train_time/60:.1f}min)")

    return results
