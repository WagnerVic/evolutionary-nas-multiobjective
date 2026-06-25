"""Avaliador de genótipos: treina a CNN e retorna os 3 objetivos brutos.

Também fornece a função de warm-up para estimar f2_ref e f3_ref antes da busca.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.genotype import decode, random_genotype, N_GENES
from src.model import build_model, count_params, count_macs
from src.constants import N_WARMUP, WARMUP_SEED, F2_REF_FALLBACK, F3_REF_FALLBACK


# ──────────────────────────────────────────────────────────────────────────────
# Reprodutibilidade (R6)
# ──────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    """Fixa semente em torch, numpy e random para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_deterministic() -> None:
    """Configura PyTorch para execução determinística."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Warm-up de normalização
# ──────────────────────────────────────────────────────────────────────────────

def compute_normalization_refs(n_samples: int = N_WARMUP, seed: int = WARMUP_SEED) -> tuple[float, float]:
    """Amostra arquiteturas aleatórias e calcula f2_ref e f3_ref como percentil 95.

    Não treina modelos — apenas constrói e conta parâmetros/MACs.

    Returns:
        (f2_ref, f3_ref): valores de referência para normalização.
    """
    rng = np.random.default_rng(seed)
    f2_values = []
    f3_values = []

    for _ in range(n_samples):
        g = random_genotype(rng)
        cfg = decode(g)
        model = build_model(cfg)
        f2_values.append(count_params(model))
        f3_values.append(count_macs(model))

    f2_ref = float(np.percentile(f2_values, 95))
    f3_ref = float(np.percentile(f3_values, 95))

    if f2_ref <= 0:
        f2_ref = F2_REF_FALLBACK
    if f3_ref <= 0:
        f3_ref = F3_REF_FALLBACK

    return f2_ref, f3_ref


# ──────────────────────────────────────────────────────────────────────────────
# Avaliação principal
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    genotype: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    seed: int,
    n_epochs: int = 4,
    track_energy: bool = False,
) -> tuple[float, float, float]:
    """Treina a CNN e retorna objetivos brutos (f1, f2, f3).

    Args:
        genotype: vetor de 20 genes (índices inteiros).
        train_loader: DataLoader do conjunto de treino.
        val_loader: DataLoader do conjunto de validação (ou teste).
        device: 'cuda' ou 'cpu'.
        seed: semente para reprodutibilidade do treinamento.
        n_epochs: número de épocas de treinamento.
        track_energy: se True, tenta rastrear energia com CodeCarbon.

    Returns:
        (f1, f2, f3) onde:
            f1 = 1 - acurácia na validação
            f2 = número de parâmetros treináveis
            f3 = MACs (multiply-accumulate operations)
    """
    _seed_everything(seed)

    cfg = decode(genotype)
    model = build_model(cfg).to(device)

    f2 = count_params(model)
    f3 = count_macs(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    tracker: Optional[object] = None
    if track_energy:
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(log_level="error", save_to_file=False)
            tracker.start()  # type: ignore[union-attr]
        except Exception:
            tracker = None

    # ── Treino ────────────────────────────────────────────────────────────────
    model.train()
    for _ in range(n_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # ── Validação ─────────────────────────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = correct / total
    f1 = 1.0 - val_acc

    if tracker is not None:
        try:
            tracker.stop()  # type: ignore[union-attr]
        except Exception:
            pass

    return f1, float(f2), float(f3)
