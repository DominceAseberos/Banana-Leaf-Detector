import os
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

TRAIN_CLASS_MAP = {"Healthy_leaf": 0, "Diseased_Leaf": 1, "Non_leaf": 2}
OUTPUT_MAP = {0: "Healthy Leaf", 1: "Unhealthy Leaf", 2: "Non-Leaf"}

train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=2):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Training data not found at {data_dir}. "
            "Download from: https://drive.google.com/drive/folders/1mng06d0Y_U4hC7WM5hnbBNbuC5ohulcq"
        )

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = full_dataset.classes
    targets = np.array(full_dataset.targets)
    n_samples = len(targets)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(n_samples), targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(
        datasets.ImageFolder(data_dir, transform=val_transform), val_idx
    )

    class_counts = np.bincount(targets[train_idx])
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets[train_idx]]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, class_names
