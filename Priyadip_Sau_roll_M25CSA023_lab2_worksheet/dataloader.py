"""
Custom CIFAR-10 Dataset and DataLoader with  Augmentations
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import numpy as np

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10CustomDataset(Dataset):
    """
    Custom CIFAR-10 Dataset wrapper with configurable augmentations.
    
    Args:
        root: Root directory for dataset
        train: If True, creates dataset from training set
        transform: Optional transform to be applied on a sample
        download: If True, downloads the dataset
    """
    
    def __init__(self, root='./data', train=True, transform=None, download=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None
        )
        self.transform = transform
        self.train = train
        
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label):
        return self.classes[label]


def get_transforms(train=True):
    """
    Get data transforms for training or validation.
    
    Training transforms include:
    - RandomCrop with padding
    - RandomHorizontalFlip
    - AutoAugment (CIFAR-10 policy)
    - Normalization
    
    Validation transforms include:
    - Normalization only
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    
    return transform


def create_dataloaders(data_dir='./data', batch_size=128, num_workers=4):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    train_dataset = CIFAR10CustomDataset(
        root=data_dir,
        train=True,
        transform=get_transforms(train=True),
        download=True
    )
    
    val_dataset = CIFAR10CustomDataset(
        root=data_dir,
        train=False,
        transform=get_transforms(train=False),
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset, val_dataset


def denormalize(tensor):
    """Denormalize a tensor for visualization."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return tensor * std + mean


if __name__ == "__main__":
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders()
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {[train_dataset.get_class_name(l.item()) for l in labels[:5]]}")