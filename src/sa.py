"""Simulated Annealing para NAS leve multiobjetivo.

Busca local com:
  - Vizinhança por perturbação de 1 gene
  - Soma ponderada dos objetivos normalizados
  - Critério de Metropolis
  - Resfriamento geométrico (alpha=0.98)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.genotype import random_genotype, mutate
from src.evaluator import evaluate


@dataclass
class SAResult:
    """Resultado de uma execução do SA."""

    best_genotype: np.ndarray
    best_objectives: tuple[float, float, float]  # (f1, f2, f3) brutos
    history: pd.DataFrame  # colunas: iter, temperature, fitness, f1, f2, f3
    seed: int
    weights: tuple[float, float, float]
    runtime_s: float


class SimulatedAnnealing:
    """Simulated Annealing com soma ponderada para NAS multiobjetivo."""

    def __init__(
        self,
        weights: tuple[float, float, float],
        f2_ref: float,
        f3_ref: float,
        T0: float = 1.0,
        alpha: float = 0.98,
        n_iter: int = 150,
    ):
        self.weights = weights
        self.f2_ref = f2_ref
        self.f3_ref = f3_ref
        self.T0 = T0
        self.alpha = alpha
        self.n_iter = n_iter

    def _fitness(self, f1: float, f2: float, f3: float) -> float:
        """Soma ponderada com normalização por referências do warm-up."""
        w1, w2, w3 = self.weights
        return w1 * f1 + w2 * (f2 / self.f2_ref) + w3 * (f3 / self.f3_ref)

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        seed: int,
        n_epochs: int = 4,
    ) -> SAResult:
        """Executa o SA e retorna o resultado completo.

        Args:
            train_loader: DataLoader de treino.
            val_loader: DataLoader de validação.
            device: 'cuda' ou 'cpu'.
            seed: semente para o SA e para as avaliações.
            n_epochs: épocas de treinamento por avaliação (proxy).

        Returns:
            SAResult com melhor genótipo, objetivos, histórico e tempo.
        """
        t_start = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Solução inicial
        s = random_genotype(rng)
        f1, f2, f3 = evaluate(s, train_loader, val_loader, device, seed, n_epochs)
        fitness_s = self._fitness(f1, f2, f3)

        best_s = s.copy()
        best_fitness = fitness_s
        best_objs = (f1, f2, f3)

        T = self.T0
        history_rows: list[dict] = []

        for k in range(1, self.n_iter + 1):
            # Gerar vizinho
            s_new = mutate(s, rng)

            # Avaliar vizinho (semente diferente por iteração para evitar bias)
            eval_seed = seed * 10000 + k
            f1_new, f2_new, f3_new = evaluate(
                s_new, train_loader, val_loader, device, eval_seed, n_epochs
            )
            fitness_new = self._fitness(f1_new, f2_new, f3_new)

            # Critério de Metropolis
            delta = fitness_new - fitness_s
            if delta < 0 or rng.random() < math.exp(-delta / T):
                s = s_new
                f1, f2, f3 = f1_new, f2_new, f3_new
                fitness_s = fitness_new

            # Tracking do melhor (usa cache, sem recálculo)
            if fitness_s < best_fitness:
                best_s = s.copy()
                best_fitness = fitness_s
                best_objs = (f1, f2, f3)

            # Resfriamento
            T = T * self.alpha

            # Registra estado ATUAL de s (pós-decisão)
            history_rows.append({
                "iter": k,
                "temperature": T,
                "fitness": fitness_s,
                "f1": f1,
                "f2": f2,
                "f3": f3,
            })

        runtime = time.perf_counter() - t_start

        return SAResult(
            best_genotype=best_s,
            best_objectives=best_objs,
            history=pd.DataFrame(history_rows),
            seed=seed,
            weights=self.weights,
            runtime_s=runtime,
        )
