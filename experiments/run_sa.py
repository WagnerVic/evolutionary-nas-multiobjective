"""Experimento SA: 10 sementes × 5 vetores de peso + avaliação final no teste.

Uso:
    python -m experiments.run_sa --device cuda
    python -m experiments.run_sa --epochs-proxy 4 --n-iter 150 --device cuda
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

from src.constants import SPLIT_SEED, WEIGHT_VECTORS
from src.evaluator import setup_deterministic, compute_normalization_refs, evaluate
from src.sa import SimulatedAnnealing


def get_data_loaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Carrega Fashion-MNIST com split determinístico 50k/10k/10k."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = FashionMNIST(root=root, train=True, download=False, transform=transform)
    test_set = FashionMNIST(root=root, train=False, download=False, transform=transform)

    # Split determinístico usando SPLIT_SEED
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento SA para NAS multiobjetivo")
    parser.add_argument("--epochs-proxy", type=int, default=4, help="Épocas durante a busca (proxy)")
    parser.add_argument("--epochs-final", type=int, default=15, help="Épocas na avaliação final no teste")
    parser.add_argument("--n-iter", type=int, default=150, help="Iterações do SA por execução")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Lista de sementes")
    parser.add_argument("--root", type=str, default="data", help="Pasta dos datasets")
    parser.add_argument("--results-dir", type=str, default="results", help="Pasta para salvar resultados")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    setup_deterministic()

    print(f"[config] device={args.device}, epochs_proxy={args.epochs_proxy}, "
          f"epochs_final={args.epochs_final}, n_iter={args.n_iter}")
    print(f"[config] seeds={args.seeds}, weights={len(WEIGHT_VECTORS)} vetores")

    # Carregar dados
    print("[data] Carregando Fashion-MNIST...")
    train_loader, val_loader, test_loader = get_data_loaders(root=args.root)

    # Warm-up de normalização
    print("[warmup] Calculando f2_ref e f3_ref...")
    f2_ref, f3_ref = compute_normalization_refs()
    print(f"[warmup] f2_ref={f2_ref:.0f}, f3_ref={f3_ref:.0f}")

    # ── Execuções do SA ───────────────────────────────────────────────────────
    raw_rows: list[dict] = []
    history_rows: list[dict] = []
    best_genotypes: list[tuple[int, float, float, float, np.ndarray]] = []

    total_runs = len(args.seeds) * len(WEIGHT_VECTORS)
    run_idx = 0

    for seed in args.seeds:
        for w1, w2, w3 in WEIGHT_VECTORS:
            run_idx += 1
            print(f"\n[SA {run_idx}/{total_runs}] seed={seed}, weights=({w1:.2f},{w2:.2f},{w3:.2f})")

            sa = SimulatedAnnealing(
                weights=(w1, w2, w3),
                f2_ref=f2_ref,
                f3_ref=f3_ref,
                T0=1.0,
                alpha=0.98,
                n_iter=args.n_iter,
            )

            result = sa.run(
                train_loader=train_loader,
                val_loader=val_loader,
                device=args.device,
                seed=seed,
                n_epochs=args.epochs_proxy,
            )

            f1_best, f2_best, f3_best = result.best_objectives
            print(f"  -> best: f1={f1_best:.4f}, f2={f2_best:.0f}, f3={f3_best:.0f} "
                  f"({result.runtime_s:.1f}s)")

            raw_rows.append({
                "seed": seed,
                "w1": w1, "w2": w2, "w3": w3,
                "f1": f1_best, "f2": f2_best, "f3": f3_best,
                "runtime_s": result.runtime_s,
            })

            # Guardar genótipo para avaliação final no teste
            best_genotypes.append((seed, w1, w2, w3, result.best_genotype.copy()))

            # Histórico com identificação da run
            hist = result.history.copy()
            hist["seed"] = seed
            hist["w1"] = w1
            hist["w2"] = w2
            hist["w3"] = w3
            history_rows.append(hist)

    # Salvar resultados brutos
    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(results_dir / "sa_raw.csv", index=False)
    print(f"\n[salvo] {results_dir / 'sa_raw.csv'}")

    df_history = pd.concat(history_rows, ignore_index=True)
    col_order = ["seed", "w1", "w2", "w3", "iter", "temperature", "fitness", "f1", "f2", "f3"]
    df_history = df_history[col_order]
    df_history.to_csv(results_dir / "sa_history.csv", index=False)
    print(f"[salvo] {results_dir / 'sa_history.csv'}")

    # ── Avaliação final no conjunto de teste ──────────────────────────────────
    print(f"\n[teste] Avaliação final dos {len(best_genotypes)} melhores genótipos "
          f"({args.epochs_final} épocas)...")

    test_rows: list[dict] = []

    for seed, w1, w2, w3, genotype in tqdm(best_genotypes, desc="test eval"):
        f1_test, f2_test, f3_test = evaluate(
            genotype,
            train_loader,
            test_loader,
            device=args.device,
            seed=seed,
            n_epochs=args.epochs_final,
        )
        test_acc = 1.0 - f1_test

        test_rows.append({
            "seed": seed,
            "w1": w1, "w2": w2, "w3": w3,
            "test_acc": test_acc,
            "f2": f2_test, "f3": f3_test,
        })

    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(results_dir / "sa_test_eval.csv", index=False)
    print(f"[salvo] {results_dir / 'sa_test_eval.csv'}")
    print("\n[concluído] Experimento SA finalizado.")


if __name__ == "__main__":
    main()
