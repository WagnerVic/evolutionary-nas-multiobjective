"""Baixa os datasets do projeto para a pasta data/.

Datasets (proposta P8 — NAS leve multiobjetivo):
  - Fashion-MNIST: benchmark principal (10 classes, 28x28, 70k imagens).
  - MNIST:         instância secundária para checagem de sanidade.

Os dados são baixados pelo torchvision a partir da fonte oficial, então
não é preciso conta no Kaggle nem download manual (favorece o R6).

Uso:
    python scripts/download_data.py            # baixa os dois
    python scripts/download_data.py --only fashion
    python scripts/download_data.py --root data
"""

import argparse
from pathlib import Path

from torchvision.datasets import MNIST, FashionMNIST


def baixar(root: str, only: str) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    if only in ("fashion", "ambos"):
        print(f"[download] Fashion-MNIST -> {root_path}/FashionMNIST")
        FashionMNIST(root=root, train=True, download=True)
        FashionMNIST(root=root, train=False, download=True)

    if only in ("mnist", "ambos"):
        print(f"[download] MNIST (sanity check) -> {root_path}/MNIST")
        MNIST(root=root, train=True, download=True)
        MNIST(root=root, train=False, download=True)

    print("[ok] datasets prontos em", root_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa os datasets do projeto.")
    parser.add_argument(
        "--root", default="data", help="pasta de destino (default: data)"
    )
    parser.add_argument(
        "--only",
        choices=["fashion", "mnist", "ambos"],
        default="ambos",
        help="qual dataset baixar (default: ambos)",
    )
    args = parser.parse_args()
    baixar(args.root, args.only)
