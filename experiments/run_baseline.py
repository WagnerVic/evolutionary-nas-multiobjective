"""Baseline: busca aleatória + LeNet-5 aproximada.

Usa mesmas sementes, split e warm-up que o SA para comparação justa (R4).

Uso:
    python -m experiments.run_baseline --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from src.constants import SPLIT_SEED, WEIGHT_VECTORS, LENET5_GENOTYPE
from src.evaluator import (
    setup_deterministic,
    compute_normalization_refs,
    evaluate,
)
from src.genotype import random_genotype


def get_data_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Carrega Fashion-MNIST com split determinístico 50k/10k/10k (idêntico ao run_sa)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = FashionMNIST(root=root, train=True, download=False, transform=transform)
    test_set = FashionMNIST(root=root, train=False, download=False, transform=transform)

    gen = torch.Generator().manual_seed(SPLIT_SEED)
    train_set, val_set = random_split(full_train, [50000, 10000], generator=gen)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def _fitness(
    f1: float, f2: float, f3: float,
    weights: tuple[float, float, float],
    f2_ref: float, f3_ref: float,
) -> float:
    """Soma ponderada normalizada (idêntica à usada no SA)."""
    w1, w2, w3 = weights
    return w1 * f1 + w2 * (f2 / f2_ref) + w3 * (f3 / f3_ref)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: busca aleatória + LeNet-5")
    parser.add_argument("--epochs-proxy", type=int, default=4, help="Épocas por avaliação (proxy)")
    parser.add_argument("--epochs-final", type=int, default=15, help="Épocas para LeNet-5 (avaliação completa)")
    parser.add_argument("--n-evals", type=int, default=150, help="Avaliações por semente (mesmo orçamento do SA)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_deterministic()

    print(f"[config] device={args.device}, epochs_proxy={args.epochs_proxy}, "
          f"n_evals={args.n_evals}")

    # Dados (split idêntico ao SA)
    print("[data] Carregando Fashion-MNIST...")
    train_loader, val_loader, test_loader = get_data_loaders(root=args.root)

    # Warm-up (idêntico ao SA — mesma seed, mesmos valores)
    print("[warmup] Calculando f2_ref e f3_ref...")
    f2_ref, f3_ref = compute_normalization_refs()
    print(f"[warmup] f2_ref={f2_ref:.0f}, f3_ref={f3_ref:.0f}")

    # ── Busca aleatória ───────────────────────────────────────────────────────
    raw_rows: list[dict] = []

    for seed in args.seeds:
        for w1, w2, w3 in WEIGHT_VECTORS:
            print(f"\n[random search] seed={seed}, weights=({w1:.2f},{w2:.2f},{w3:.2f})")
            rng = np.random.default_rng(seed)

            best_fitness = float("inf")
            best_objs = (1.0, 0.0, 0.0)

            for i in tqdm(range(args.n_evals), desc=f"  seed={seed}", leave=False):
                g = random_genotype(rng)
                eval_seed = seed * 10000 + i
                f1, f2, f3 = evaluate(
                    g, train_loader, val_loader,
                    device=args.device, seed=eval_seed, n_epochs=args.epochs_proxy,
                )
                fit = _fitness(f1, f2, f3, (w1, w2, w3), f2_ref, f3_ref)

                if fit < best_fitness:
                    best_fitness = fit
                    best_objs = (f1, f2, f3)

            raw_rows.append({
                "method": "random_search",
                "seed": seed,
                "w1": w1, "w2": w2, "w3": w3,
                "f1": best_objs[0], "f2": best_objs[1], "f3": best_objs[2],
            })
            print(f"  -> best: f1={best_objs[0]:.4f}, f2={best_objs[1]:.0f}, f3={best_objs[2]:.0f}")

    # ── LeNet-5 aproximada ────────────────────────────────────────────────────
    print("\n[lenet5] Avaliando LeNet-5 aproximada (avaliação completa)...")

    for seed in tqdm(args.seeds, desc="LeNet-5"):
        f1, f2, f3 = evaluate(
            LENET5_GENOTYPE, train_loader, test_loader,
            device=args.device, seed=seed, n_epochs=args.epochs_final,
        )
        test_acc = 1.0 - f1

        raw_rows.append({
            "method": "lenet5_approx",
            "seed": seed,
            "w1": 0.0, "w2": 0.0, "w3": 0.0,
            "f1": f1, "f2": f2, "f3": f3,
        })
        print(f"  seed={seed}: test_acc={test_acc:.4f}, f2={f2:.0f}, f3={f3:.0f}")

    # Salvar
    df = pd.DataFrame(raw_rows)
    df.to_csv(results_dir / "baseline_raw.csv", index=False)
    print(f"\n[salvo] {results_dir / 'baseline_raw.csv'}")
    print("[concluído] Baseline finalizado.")


if __name__ == "__main__":
    main()
