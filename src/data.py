"""Carregamento e split do Fashion-MNIST/MNIST (seção 3.3 — protocolo de avaliação).

O `test` oficial do torchvision nunca é tocado pela busca: só entra em
`evaluate_test_accuracy` (treino_eval.py), na avaliação final da fronteira
de Pareto. O split treino/validação é fixo (seed), para que SA e baseline
(quando implementados) usem exatamente a mesma partição.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

_TRANSFORM = transforms.ToTensor()


def _get_targets(dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        return _get_targets(dataset.dataset)[dataset.indices]
    return np.asarray(dataset.targets)


def _split_train_val(train_full, val_fraction: float, seed: int):
    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(train_full, [n_train, n_val], generator=generator)


def load_fashion_mnist(root: str = "data", val_fraction: float = 0.2, seed: int = 42):
    train_full = FashionMNIST(root=root, train=True, download=False, transform=_TRANSFORM)
    test_ds = FashionMNIST(root=root, train=False, download=False, transform=_TRANSFORM)
    train_ds, val_ds = _split_train_val(train_full, val_fraction, seed)
    return train_ds, val_ds, test_ds


def load_mnist(root: str = "data", val_fraction: float = 0.2, seed: int = 42):
    train_full = MNIST(root=root, train=True, download=False, transform=_TRANSFORM)
    test_ds = MNIST(root=root, train=False, download=False, transform=_TRANSFORM)
    train_ds, val_ds = _split_train_val(train_full, val_fraction, seed)
    return train_ds, val_ds, test_ds


def _stratified_subset(dataset, size: int | None, seed: int):
    """Sub-amostra estratificada por classe, para acelerar avaliações na busca."""
    if size is None or size >= len(dataset):
        return dataset

    targets = _get_targets(dataset)
    classes = np.unique(targets)
    rng = np.random.default_rng(seed)
    per_class = size // len(classes)

    chosen: list[int] = []
    for c in classes:
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        chosen.extend(class_idx[:per_class].tolist())

    remaining = size - len(chosen)
    if remaining > 0:
        leftover = np.setdiff1d(np.arange(len(dataset)), np.array(chosen))
        rng.shuffle(leftover)
        chosen.extend(leftover[:remaining].tolist())

    return Subset(dataset, chosen)


def make_loaders(
    train_ds,
    val_ds,
    batch_size: int = 128,
    train_size: int | None = None,
    val_size: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    train_subset = _stratified_subset(train_ds, train_size, seed)
    val_subset = _stratified_subset(val_ds, val_size, seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_test_loader(test_ds, batch_size: int = 256) -> DataLoader:
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
