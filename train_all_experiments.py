"""
Run all Task 1 experiments in sequence.

Phases:
  1. Baseline: VGG_A training
  2. Activation function comparison (VGG_Half proxy)
  3. Loss function & regularization (VGG_Half proxy)
  4. Architecture comparison (full models)
  5. Optimizer comparison (best architecture)

Usage:
    python train_all_experiments.py              # run all phases
    python train_all_experiments.py --phase 1    # run only phase 1
    python train_all_experiments.py --phase 2    # run only phase 2
    python train_all_experiments.py --quick      # fast test with 5 epochs
"""

import sys
import os
import torch
import time
import argparse

# Ensure working directory is pj2_task1/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout
from models.vgg_variants import VGG_Half, VGG_Quarter
from models.residual import VGG_Residual
from models.simple_cnn import SimpleCNN
from data.loaders import get_train_val_loaders, get_cifar_loader
from experiments.runner import ExperimentConfig, run_experiment
from experiments.results import save_result, load_results, generate_summary_table
from visualization.curves import plot_training_curves, plot_comparison_curves, plot_phase_summary
from visualization.interpret import plot_param_vs_accuracy
from utils.training import set_random_seeds


def get_device():
    """Detect device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_data(batch_size=64, data_root='./data/', n_items_train=-1, n_items_val=-1):
    """Create train/val/test loaders. Set n_items_train > 0 for small-scale runs."""
    if n_items_train > 0:
        # Small-scale: use subset of training data for train, separate subset for val
        train_loader = get_cifar_loader(
            root=data_root, batch_size=batch_size, train=True,
            shuffle=True, num_workers=0, use_augmentation=True,
            n_items=n_items_train
        )
        val_loader = get_cifar_loader(
            root=data_root, batch_size=batch_size, train=False,
            shuffle=False, num_workers=0, use_augmentation=False,
            n_items=n_items_val if n_items_val > 0 else 2000
        )
    else:
        # Full-scale: proper train/val split from training set
        train_loader, val_loader = get_train_val_loaders(
            root=data_root, batch_size=batch_size, val_ratio=0.1, num_workers=0
        )
    test_loader = get_cifar_loader(
        root=data_root, batch_size=batch_size, train=False,
        shuffle=False, num_workers=0, use_augmentation=False
    )
    return train_loader, val_loader, test_loader


# =====================================================================
# Phase 1: Baseline
# =====================================================================

def run_phase1_baseline(device, data_root, epochs=30, batch_size=64, n_items=-1):
    """Train VGG_A baseline."""
    print("\n" + "="*70)
    print("PHASE 1: BASELINE")
    print("="*70)

    train_loader, val_loader, test_loader = get_data(batch_size, data_root, n_items_train=n_items)

    config = ExperimentConfig(
        experiment_id='P1_VGG_A_baseline',
        phase='baseline',
        model_name='VGG_A',
        model_class=VGG_A,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name='Adam',
        lr=1e-3,
        weight_decay=0.0,
        loss_fn='CrossEntropy',
        activation='ReLU',
        use_scheduler=True,
        scheduler_type='CosineAnnealingLR',
        save_dir='./saved_models/',
    )

    result = run_experiment(config, train_loader, val_loader, test_loader, device)
    save_result(result)
    plot_training_curves(result['history'],
                         f'./figures/P1_baseline_curves.png',
                         title='VGG_A Baseline Training')
    return result


# =====================================================================
# Phase 2: Activation Functions (VGG_Half, 20 epochs)
# =====================================================================

def run_phase2_activations(device, data_root, epochs=20, batch_size=64, n_items=-1):
    """Compare different activation functions using VGG_Half."""
    print("\n" + "="*70)
    print("PHASE 2: ACTIVATION FUNCTION COMPARISON")
    print("="*70)

    activations = ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'SiLU']
    results = {}
    histories = {}

    for act in activations:
        # Build VGG_Half with custom activation
        from models.vgg_variants import VGG_Half as VH
        import torch.nn as nn

        act_map = {
            'ReLU': nn.ReLU,
            'LeakyReLU': lambda: nn.LeakyReLU(0.01),
            'ELU': nn.ELU,
            'GELU': nn.GELU,
            'SiLU': nn.SiLU,
        }

        class VGG_Half_Act(VH):
            def __init__(self, activation_cls):
                super().__init__(init_weights=False)
                # Replace activations in features
                new_features = []
                for module in self.features:
                    if isinstance(module, nn.ReLU):
                        new_features.append(activation_cls())
                    else:
                        new_features.append(module)
                self.features = nn.Sequential(*new_features)
                # Replace in classifier
                new_classifier = []
                for module in self.classifier:
                    if isinstance(module, nn.ReLU):
                        new_classifier.append(activation_cls())
                    else:
                        new_classifier.append(module)
                self.classifier = nn.Sequential(*new_classifier)
                self._init_weights()

        model_cls = lambda: VGG_Half_Act(act_map[act])

        train_loader, val_loader, test_loader = get_data(batch_size, data_root, n_items_train=n_items)

        config = ExperimentConfig(
            experiment_id=f'P2_act_{act}',
            phase='activation',
            model_name=f'VGG_Half_{act}',
            model_class=model_cls,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name='Adam',
            lr=1e-3,
            loss_fn='CrossEntropy',
            activation=act,
            use_scheduler=True,
            save_dir='./saved_models/',
        )

        result = run_experiment(config, train_loader, val_loader, test_loader, device)
        save_result(result)
        results[act] = result
        histories[act] = result['history']

    # Plot comparison
    plot_comparison_curves(histories, './figures/P2_activations_val_acc.png',
                           metric='val_acc', title='Activation Function Comparison')
    plot_comparison_curves(histories, './figures/P2_activations_val_loss.png',
                           metric='val_loss', title='Activation Function Comparison')
    plot_phase_summary(
        [results[a] for a in activations], 'activation', 'test_acc',
        activations, './figures/P2_activations_bar.png',
        title='Test Accuracy by Activation Function', ylabel='Test Accuracy'
    )

    # Find best activation
    best_act = max(results, key=lambda a: results[a]['test_acc'])
    print(f"\nBest activation: {best_act} (Test Acc: {results[best_act]['test_acc']:.4f})")
    return best_act, results[best_act]


# =====================================================================
# Phase 3: Loss Functions & Regularization (VGG_Half, 20 epochs)
# =====================================================================

def run_phase3_loss(device, data_root, best_activation='GELU', epochs=20, batch_size=64, n_items=-1):
    """Compare loss functions and regularization using VGG_Half."""
    print("\n" + "="*70)
    print("PHASE 3: LOSS FUNCTION & REGULARIZATION")
    print("="*70)

    import torch.nn as nn

    act_map = {
        'ReLU': nn.ReLU, 'LeakyReLU': lambda: nn.LeakyReLU(0.01),
        'ELU': nn.ELU, 'GELU': nn.GELU, 'SiLU': nn.SiLU,
    }

    from models.vgg_variants import VGG_Half as VH

    class VGG_Half_Act(VH):
        def __init__(self, activation_cls):
            super().__init__(init_weights=False)
            new_features = []
            for m in self.features:
                new_features.append(activation_cls() if isinstance(m, nn.ReLU) else m)
            self.features = nn.Sequential(*new_features)
            new_classifier = []
            for m in self.classifier:
                new_classifier.append(activation_cls() if isinstance(m, nn.ReLU) else m)
            self.classifier = nn.Sequential(*new_classifier)
            self._init_weights()

    # Loss configs: (id_suffix, loss_name, wd, label_smoothing, l1_lambda)
    loss_configs = [
        ('CE', 'CrossEntropy', 0.0, 0.0, 0.0),
        ('CE_L2_1e4', 'CrossEntropy', 1e-4, 0.0, 0.0),
        ('CE_L2_5e4', 'CrossEntropy', 5e-4, 0.0, 0.0),
        ('CE_L2_1e3', 'CrossEntropy', 1e-3, 0.0, 0.0),
        ('CE_LS_01', 'CrossEntropy', 0.0, 0.1, 0.0),
        ('CE_L1_1e5', 'CrossEntropy', 0.0, 0.0, 1e-5),
    ]

    results = {}
    histories = {}

    act_fn = act_map.get(best_activation, nn.GELU)

    for suffix, loss_name, wd, ls, l1 in loss_configs:
        model_cls = lambda: VGG_Half_Act(act_fn)

        train_loader, val_loader, test_loader = get_data(batch_size, data_root, n_items_train=n_items)

        config = ExperimentConfig(
            experiment_id=f'P3_loss_{suffix}',
            phase='loss',
            model_name=f'VGG_Half_{best_activation}',
            model_class=model_cls,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name='Adam',
            lr=1e-3,
            weight_decay=wd,
            loss_fn=loss_name,
            label_smoothing=ls,
            l1_lambda=l1,
            activation=best_activation,
            use_scheduler=True,
            save_dir='./saved_models/',
        )

        result = run_experiment(config, train_loader, val_loader, test_loader, device)
        save_result(result)
        results[suffix] = result
        histories[suffix] = result['history']

    # Plot comparison
    plot_comparison_curves(histories, './figures/P3_loss_val_acc.png',
                           metric='val_acc', title='Loss Function & Regularization')
    plot_phase_summary(
        [results[c[0]] for c in loss_configs], 'experiment_id', 'test_acc',
        [c[0] for c in loss_configs], './figures/P3_loss_bar.png',
        title='Test Accuracy by Loss/Regularization', ylabel='Test Accuracy'
    )

    best_loss = max(results, key=lambda k: results[k]['test_acc'])
    print(f"\nBest loss config: {best_loss} (Test Acc: {results[best_loss]['test_acc']:.4f})")
    return best_loss, results[best_loss]


# =====================================================================
# Phase 4: Architecture Comparison (full models, 30 epochs)
# =====================================================================

def run_phase4_architectures(device, data_root, best_activation='GELU',
                             best_loss_wd=0.0, epochs=30, batch_size=64, n_items=-1):
    """Compare all architectures at full scale."""
    print("\n" + "="*70)
    print("PHASE 4: ARCHITECTURE COMPARISON")
    print("="*70)

    model_configs = [
        ('VGG_A', VGG_A, 64),
        ('VGG_A_BatchNorm', VGG_A_BatchNorm, 64),
        ('VGG_A_Dropout', VGG_A_Dropout, 64),
        ('VGG_Residual', VGG_Residual, 64),
        ('VGG_Half', VGG_Half, 128),
        ('VGG_Quarter', VGG_Quarter, 128),
        ('SimpleCNN', SimpleCNN, 128),
    ]

    results = {}
    histories = {}

    for model_name, model_cls, bs in model_configs:
        train_loader, val_loader, test_loader = get_data(bs, data_root, n_items_train=n_items)

        config = ExperimentConfig(
            experiment_id=f'P4_arch_{model_name}',
            phase='architecture',
            model_name=model_name,
            model_class=model_cls,
            epochs=epochs,
            batch_size=bs,
            optimizer_name='Adam',
            lr=1e-3,
            weight_decay=best_loss_wd,
            loss_fn='CrossEntropy',
            activation=best_activation,
            use_scheduler=True,
            save_dir='./saved_models/',
        )

        result = run_experiment(config, train_loader, val_loader, test_loader, device)
        save_result(result)
        results[model_name] = result
        histories[model_name] = result['history']

    # Plot comparison
    plot_comparison_curves(histories, './figures/P4_arch_val_acc.png',
                           metric='val_acc', title='Architecture Comparison')
    plot_param_vs_accuracy(
        [results[m[0]] for m in model_configs],
        './figures/P4_param_vs_acc.png'
    )

    best_arch = max(results, key=lambda k: results[k]['test_acc'])
    print(f"\nBest architecture: {best_arch} (Test Acc: {results[best_arch]['test_acc']:.4f})")
    return best_arch, results[best_arch]


# =====================================================================
# Phase 5: Optimizer Comparison (best architecture, 30 epochs)
# =====================================================================

def run_phase5_optimizers(device, data_root, best_model_cls=VGG_A_BatchNorm,
                          best_model_name='VGG_A_BatchNorm',
                          best_activation='GELU', best_loss_wd=0.0,
                          epochs=30, batch_size=64, n_items=-1):
    """Compare different optimizers."""
    print("\n" + "="*70)
    print("PHASE 5: OPTIMIZER COMPARISON")
    print("="*70)

    opt_configs = [
        ('SGD_mom0.9', 'SGD', 0.01, 0.0, 0.9),
        ('Adam', 'Adam', 1e-3, 0.0, 0.0),
        ('AdamW', 'AdamW', 1e-3, 5e-4, 0.0),
        ('RMSprop', 'RMSprop', 1e-3, 0.0, 0.0),
    ]

    results = {}
    histories = {}

    for suffix, opt_name, lr, wd, mom in opt_configs:
        train_loader, val_loader, test_loader = get_data(batch_size, data_root, n_items_train=n_items)

        config = ExperimentConfig(
            experiment_id=f'P5_opt_{suffix}',
            phase='optimizer',
            model_name=best_model_name,
            model_class=best_model_cls,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=opt_name,
            lr=lr,
            weight_decay=wd,
            momentum=mom,
            loss_fn='CrossEntropy',
            activation=best_activation,
            use_scheduler=True,
            save_dir='./saved_models/',
        )

        result = run_experiment(config, train_loader, val_loader, test_loader, device)
        save_result(result)
        results[suffix] = result
        histories[suffix] = result['history']

    # Plot comparison
    plot_comparison_curves(histories, './figures/P5_optimizers_val_acc.png',
                           metric='val_acc', title='Optimizer Comparison')
    plot_comparison_curves(histories, './figures/P5_optimizers_val_loss.png',
                           metric='val_loss', title='Optimizer Convergence Speed')
    plot_phase_summary(
        [results[c[0]] for c in opt_configs], 'optimizer_name', 'test_acc',
        [c[0] for c in opt_configs], './figures/P5_optimizers_bar.png',
        title='Test Accuracy by Optimizer', ylabel='Test Accuracy'
    )

    best_opt = max(results, key=lambda k: results[k]['test_acc'])
    print(f"\nBest optimizer: {best_opt} (Test Acc: {results[best_opt]['test_acc']:.4f})")
    return best_opt, results[best_opt]


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Task 1 experiments')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run only a specific phase')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced epochs')
    parser.add_argument('--small', action='store_true',
                        help='Use 15000 training samples (fast, for verification)')
    parser.add_argument('--data-root', type=str, default='./data/',
                        help='CIFAR-10 data root')
    args = parser.parse_args()

    # Create output directories
    os.makedirs('./figures/', exist_ok=True)
    os.makedirs('./saved_models/', exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    set_random_seeds(42, str(device))

    # Determine data scale and epochs
    if args.small:
        n_items = 15000
        phase_epochs = 5   # 5 epochs per experiment
        print(f"[SMALL MODE] {n_items} training samples, {phase_epochs} epochs/experiment")
    elif args.quick:
        n_items = -1
        phase_epochs = 10
        print(f"[QUICK MODE] Full data, {phase_epochs} epochs/experiment")
    else:
        n_items = -1
        phase_epochs = 0  # 0 means use default: 30/20 per phase

    start_time = time.time()

    if args.phase is None or args.phase == 1:
        run_phase1_baseline(device, args.data_root,
                           epochs=phase_epochs if phase_epochs > 0 else 30,
                           batch_size=64, n_items=n_items)
        print("\nPhase 1 complete.\n")

    if args.phase is None or args.phase == 2:
        best_act, _ = run_phase2_activations(device, args.data_root,
                                             epochs=phase_epochs if phase_epochs > 0 else 20,
                                             batch_size=64, n_items=n_items)
        print("\nPhase 2 complete.\n")
    else:
        best_act = 'GELU'

    if args.phase is None or args.phase == 3:
        best_loss, _ = run_phase3_loss(device, args.data_root,
                                       best_activation=best_act,
                                       epochs=phase_epochs if phase_epochs > 0 else 20,
                                       batch_size=64, n_items=n_items)
        print("\nPhase 3 complete.\n")
        # Determine best weight decay from results
        best_loss_wd = 0.0
    else:
        best_loss_wd = 0.0

    if args.phase is None or args.phase == 4:
        best_arch, arch_result = run_phase4_architectures(
            device, args.data_root, best_activation=best_act,
            best_loss_wd=best_loss_wd,
            epochs=phase_epochs if phase_epochs > 0 else 30,
            batch_size=64, n_items=n_items
        )
        print("\nPhase 4 complete.\n")

        # Determine best model class for phase 5
        model_map = {
            'VGG_A': VGG_A, 'VGG_A_BatchNorm': VGG_A_BatchNorm,
            'VGG_A_Dropout': VGG_A_Dropout, 'VGG_Residual': VGG_Residual,
            'VGG_Half': VGG_Half, 'VGG_Quarter': VGG_Quarter,
            'SimpleCNN': SimpleCNN,
        }
        best_model_cls = model_map.get(best_arch, VGG_A_BatchNorm)
    else:
        best_arch = 'VGG_A_BatchNorm'
        best_model_cls = VGG_A_BatchNorm

    if args.phase is None or args.phase == 5:
        run_phase5_optimizers(device, args.data_root,
                             best_model_cls=best_model_cls,
                             best_model_name=best_arch,
                             best_activation=best_act,
                             best_loss_wd=best_loss_wd,
                             epochs=phase_epochs if phase_epochs > 0 else 30,
                             batch_size=64, n_items=n_items)
        print("\nPhase 5 complete.\n")

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"ALL EXPERIMENTS COMPLETE ({total_time/60:.1f} min)")
    print("="*70)
    print(generate_summary_table())


if __name__ == '__main__':
    main()
