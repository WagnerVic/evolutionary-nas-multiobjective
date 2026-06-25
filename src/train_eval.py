"""Treino e avaliação determinísticos de um genótipo (seção 3.3).

Mesma seed e mesmo nº de épocas para todo indivíduo: dado um genótipo,
`evaluate` é uma função determinística pura (sem aleatoriedade entre
chamadas), então não é preciso cache de pesos treinados — a fronteira de
Pareto final pode ser re-avaliada no conjunto de teste chamando
`evaluate_test_accuracy` com o mesmo protocolo.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from src.genotype import Genotype, repair
from src.model import build_model, count_macs, count_params


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit(g: Genotype, train_loader, epochs: int, device: str, seed: int) -> nn.Module:
    _set_seed(seed)
    model = build_model(g).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=g.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def _accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate(
    genotype: Genotype,
    train_loader,
    val_loader,
    epochs: int,
    device: str,
    seed: int,
) -> dict:
    """Treina o genótipo (reparado) e mede f1 (erro de validação), f2 (params), f3 (MACs)."""
    g = repair(genotype)
    model = _fit(g, train_loader, epochs, device, seed)
    val_acc = _accuracy(model, val_loader, device)
    params = count_params(model)
    macs = count_macs(g)
    return {
        "genotype": g,
        "f1": 1.0 - val_acc,
        "f2": params,
        "f3": macs,
        "val_acc": val_acc,
        "params": params,
        "macs": macs,
    }


def evaluate_test_accuracy(
    genotype: Genotype,
    train_loader,
    test_loader,
    epochs: int,
    device: str,
    seed: int,
) -> float:
    """Repete o mesmo treino determinístico e mede acurácia no teste (fora da busca)."""
    g = repair(genotype)
    model = _fit(g, train_loader, epochs, device, seed)
    return _accuracy(model, test_loader, device)
