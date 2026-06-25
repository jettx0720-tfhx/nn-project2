"""
CIFAR-10 Data Loader with data augmentation.
Fixed version from VGG_BatchNorm/data/loaders.py.

Usage:
    train_loader = get_cifar_loader(train=True, batch_size=128)
    test_loader  = get_cifar_loader(train=False, batch_size=128)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os


class PartialDataset(Dataset):
    """Wrapper to use only the first n_items of a dataset."""

    def __init__(self, dataset, n_items=10):
        super().__init__()
        self.dataset = dataset
        self.n_items = min(n_items, len(dataset))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.n_items


def get_cifar_loader(root='./data/', batch_size=128, train=True,
                     shuffle=True, num_workers=0, n_items=-1,
                     use_augmentation=True):
    """
    Get CIFAR-10 DataLoader.

    Args:
        root: data root directory
        batch_size: batch size
        train: True for training set, False for test set
        shuffle: whether to shuffle the data
        num_workers: number of worker processes
        n_items: if > 0, only use first n_items samples
        use_augmentation: if True, apply data augmentation for training

    Returns:
        DataLoader instance
    """
    if train and use_augmentation:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=transform
    )

    if n_items > 0:
        dataset = PartialDataset(dataset, n_items=n_items)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False
    )
    return loader


def get_train_val_loaders(root='./data/', batch_size=128,
                          val_ratio=0.1, num_workers=0):
    """
    Create train/val split from CIFAR-10 training set.

    Args:
        root: data root directory
        batch_size: batch size
        val_ratio: ratio of training data to use for validation
        num_workers: number of worker processes

    Returns:
        train_loader, val_loader
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset without transform to split indices
    full_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=None
    )

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    indices = list(range(len(full_dataset)))

    train_idx, val_idx = random_split(
        indices, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    class CIFARSubset(Dataset):
        """Subset with its own transform applied on __getitem__."""
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = list(indices)
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.dataset[self.indices[idx]]
            return self.transform(x), y

        def __len__(self):
            return len(self.indices)

    train_dataset = CIFARSubset(full_dataset, train_idx, transform_train)
    val_dataset = CIFARSubset(full_dataset, val_idx, transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test: load one batch and visualize
    loader = get_cifar_loader(root='../data/', batch_size=8, train=True,
                              use_augmentation=False)
    images, labels = next(iter(loader))

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).clamp(0, 1)
        ax.imshow(img)
        ax.set_title(classes[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_cifar.png')
    print(f"Sample saved. Batch shape: {images.shape}, Labels: {labels}")
