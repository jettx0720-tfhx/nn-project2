"""
Core training and evaluation functions for Task 1 experiments.
"""

import torch
import torch.nn as nn
import time
import os


def set_random_seeds(seed_value=42, device='cpu'):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device,
                    l1_lambda=0, l2_lambda=0):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)

        # Optional L1 regularization (manual)
        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss, accuracy, predictions, and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        running_loss += loss.item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += x.size(0)

        all_preds.append(pred.argmax(1).cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return running_loss / total, correct / total, all_preds, all_labels


def train_full(model, train_loader, val_loader, test_loader,
               optimizer, criterion, scheduler, device,
               epochs=30, save_path=None, verbose=True,
               l1_lambda=0):
    """
    Full training loop with validation and checkpointing.

    Returns:
        history: dict with train_loss, train_acc, val_loss, val_acc per epoch
        test_loss, test_acc, test_preds, test_labels
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, l1_lambda
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler step (for CosineAnnealingLR: step per epoch)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # Save best model
        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        if verbose:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {lr:.6f} | Time: {epoch_time:.1f}s")

    # Load best model for test evaluation
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        # Keep last epoch model
        pass

    # Test evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    if verbose:
        print(f"\nBest epoch: {best_epoch} | Best Val Acc: {best_val_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return history, test_loss, test_acc, test_preds, test_labels, best_val_acc
