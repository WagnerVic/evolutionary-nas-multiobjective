"""CLI para rodar a busca NSGA-II (Checkpoint 2) e salvar os resultados.

Uso:
    python experiments/run_nsga2.py
    python experiments/run_nsga2.py --pop-size 40 --n-gen 30 --epochs 15 \
        --train-size -1 --val-size -1

Os defaults usam um subconjunto pequeno do Fashion-MNIST e um orçamento
modesto de gerações/épocas para que a busca completa rode em poucos
minutos em CPU ("viável em laptop comum" — justificativa da proposta).
Para uma rodada "para valer", aumente --pop-size/--n-gen/--epochs e use
--train-size -1 --val-size -1 (dataset completo).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data import load_fashion_mnist, make_loaders, make_test_loader
from src.genotype import MAX_BLOCKS, PARAM_CEILING, decode, repair
from src.nsga2_search import NASProblem, run
from src.train_eval import evaluate_test_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Busca NSGA-II para NAS leve multiobjetivo (Fashion-MNIST).")
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--n-gen", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4000, help="-1 para usar todo o treino")
    parser.add_argument("--val-size", type=int, default=1000, help="-1 para usar toda a validação")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--param-ceiling", type=int, default=PARAM_CEILING)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--root", default="data", help="pasta com os datasets (scripts/download_data.py)")
    parser.add_argument("--out", default="results/nsga2")
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _none_if_negative(n: int) -> int | None:
    return None if n < 0 else n


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_pareto_rows(res, train_loader, test_loader, epochs, device, seed) -> list[dict]:
    """Decodifica a fronteira de Pareto final e mede a acurácia no teste (3.3)."""
    xs = res.X
    xs = [xs] if isinstance(xs, dict) else list(xs)
    fs = np.atleast_2d(res.F)

    rows = []
    for x, f in zip(xs, fs):
        g = repair(decode(x))
        test_acc = evaluate_test_accuracy(g, train_loader, test_loader, epochs, device, seed)
        rows.append(
            {
                "l": g.l,
                **{f"f{i}": g.filters[i] for i in range(MAX_BLOCKS)},
                **{f"k{i}": g.kernels[i] for i in range(MAX_BLOCKS)},
                **{f"p{i}": g.pools[i] for i in range(MAX_BLOCKS)},
                **{f"a{i}": g.acts[i] for i in range(MAX_BLOCKS)},
                "dense_units": g.dense_units,
                "dropout": g.dropout,
                "lr": g.lr,
                "f1_val_error": float(f[0]),
                "f2_params": int(round(f[1])),
                "f3_macs": int(round(f[2])),
                "val_acc": 1.0 - float(f[0]),
                "test_acc": test_acc,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    train_ds, val_ds, test_ds = load_fashion_mnist(root=args.root, val_fraction=args.val_fraction, seed=args.seed)
    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size,
        train_size=_none_if_negative(args.train_size),
        val_size=_none_if_negative(args.val_size),
        seed=args.seed,
    )
    test_loader = make_test_loader(test_ds)

    problem = NASProblem(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
        param_ceiling=args.param_ceiling,
    )

    print(f"[nsga2] device={device} pop_size={args.pop_size} n_gen={args.n_gen} epochs={args.epochs}")
    start = time.time()
    res = run(problem, pop_size=args.pop_size, n_gen=args.n_gen, seed=args.seed)
    elapsed = time.time() - start

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "history.csv", problem.history)
    pareto_rows = build_pareto_rows(res, train_loader, test_loader, args.epochs, device, args.seed)
    write_csv(out_dir / "pareto_front.csv", pareto_rows)

    config = {**vars(args), "device": device, "elapsed_seconds": elapsed, "n_eval": len(problem.history)}
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[ok] busca concluída em {elapsed:.1f}s — {len(problem.history)} avaliações")
    print(f"[ok] fronteira de Pareto: {len(pareto_rows)} soluções não-dominadas")
    print(f"[ok] resultados salvos em {out_dir.resolve()}")


if __name__ == "__main__":
    main()
