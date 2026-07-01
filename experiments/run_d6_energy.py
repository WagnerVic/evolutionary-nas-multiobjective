"""Diferencial D6 — Consciência energética (ética/energia como dimensão da otimização).

Mede o custo energético (kWh) e as emissões de CO₂eq de treinar arquiteturas da
**fronteira de Pareto** já encontrada pelo NSGA-II (Checkpoint 2), usando o CodeCarbon.

Ideia central: reaproveitar os genótipos da fronteira salvos em
`analysis/nsga2_results.json` (`pareto_X_seed42`) — assim medimos energia **sem
re-executar a busca** de horas. Selecionamos ~5 arquiteturas cobrindo do menor ao
maior número de parâmetros para evidenciar o trade-off *tamanho × energia*.

A codificação de 14 genes e a topologia da CNN são réplicas fiéis das usadas no
notebook oficial do CP2 (`src/nas_nsga2.ipynb`), de modo que este script roda de
forma independente (não precisa do pymoo nem do notebook).

Uso (execução decidida depois — ver README/plano):
    pip install codecarbon
    python experiments/run_d6_energy.py                 # rápido: subset 4k, 3 épocas, 5 archs
    python experiments/run_d6_energy.py --full          # dataset completo (representativo)
    python experiments/run_d6_energy.py --device cuda --n-archs 8

Limitação declarada: mede-se a energia do **treino-proxy** (poucas épocas), não do
treino completo; e o valor de CO₂eq depende da intensidade de carbono da rede local
estimada pelo CodeCarbon.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Codificação de 14 genes (réplica fiel de src/nas_nsga2.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
L_VALS = [2, 3, 4]
FILTER_VALS = [8, 16, 32, 64]
KERNEL_VALS = [3, 5, 7]
POOL_VALS = [False, True]
ACT_VALS = ["relu", "leaky"]
DENSE_VALS = [32, 64, 128]
LR_VALS = [1e-2, 1e-3, 1e-4]
DROPOUT_FIXED = 0.25


def decode_genotype(x) -> dict:
    """Vetor inteiro de 14 genes → dicionário de hiperparâmetros arquiteturais."""
    x = np.asarray(x).astype(int)
    n_blocks = L_VALS[x[0]]
    block_genes = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 4),   # bloco 3 reutiliza a1
        (9, 10, 11, 8),   # bloco 4 reutiliza a2
    ]
    blocks = []
    for i in range(n_blocks):
        fi, ki, pi, ai = block_genes[i]
        blocks.append(
            {
                "filters": FILTER_VALS[x[fi]],
                "kernel": KERNEL_VALS[x[ki]],
                "pool": POOL_VALS[x[pi]],
                "act": ACT_VALS[x[ai]],
            }
        )
    return {
        "n_blocks": n_blocks,
        "blocks": blocks,
        "dense_units": DENSE_VALS[x[12]],
        "dropout": DROPOUT_FIXED,
        "lr": LR_VALS[x[13]],
    }


def genotype_str(x) -> str:
    d = decode_genotype(x)
    blocks = "-".join(
        f"{b['filters']}x{b['kernel']}{'p' if b['pool'] else ''}{b['act'][0]}"
        for b in d["blocks"]
    )
    return f"{d['n_blocks']}blk [{blocks}] d{d['dense_units']} lr{d['lr']:.0e}"


# ──────────────────────────────────────────────────────────────────────────────
# CNN parametrizável (réplica fiel do notebook)
# ──────────────────────────────────────────────────────────────────────────────
class NASConvNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for blk in config["blocks"]:
            k = blk["kernel"]
            layers.append(nn.Conv2d(in_ch, blk["filters"], kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm2d(blk["filters"]))
            layers.append(nn.ReLU(inplace=True) if blk["act"] == "relu" else nn.LeakyReLU(0.1, inplace=True))
            if blk["pool"]:
                layers.append(nn.MaxPool2d(2))
            in_ch = blk["filters"]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, config["dense_units"]),
            nn.ReLU(inplace=True),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["dense_units"], 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_mflops(config: dict) -> float:
    """MFLOPs de inferência (torchinfo). Import tardio; retorna 0.0 se indisponível."""
    try:
        from torchinfo import summary

        model = NASConvNet(config).cpu().eval()
        stats = summary(model, input_size=(1, 1, 28, 28), verbose=0, device="cpu")
        return float(stats.total_mult_adds) * 2 / 1e6
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Dados (Fashion-MNIST) — import de torchvision tardio p/ smoke-test sem ele
# ──────────────────────────────────────────────────────────────────────────────
def make_loaders(root: str, batch_size: int, train_size, val_size, seed: int):
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    full = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(full))
    train_idx, val_idx = idx[:50000], idx[50000:]
    if train_size:
        train_idx = train_idx[:train_size]
    if val_size:
        val_idx = val_idx[:val_size]

    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=256, shuffle=False)
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Treino + medição de energia de uma arquitetura
# ──────────────────────────────────────────────────────────────────────────────
def train_and_measure(config, train_loader, val_loader, epochs, device):
    """Treina a CNN com o CodeCarbon ligado. Retorna (val_acc, energy_kWh, co2_kg)."""
    from codecarbon import EmissionsTracker  # import tardio

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = NASConvNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    tracker = EmissionsTracker(
        save_to_file=False, log_level="error", tracking_mode="machine", allow_multiple_runs=True
    )
    tracker.start()
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    co2_kg = tracker.stop() or 0.0
    try:
        energy_kwh = float(tracker.final_emissions_data.energy_consumed)
    except Exception:
        energy_kwh = 0.0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, energy_kwh, float(co2_kg)


# ──────────────────────────────────────────────────────────────────────────────
# Seleção das arquiteturas da fronteira (do menor ao maior nº de params)
# ──────────────────────────────────────────────────────────────────────────────
def select_architectures(pareto_x, n_archs: int):
    archs = []
    for x in pareto_x:
        cfg = decode_genotype(x)
        archs.append((x, cfg, count_params(NASConvNet(cfg))))
    archs.sort(key=lambda t: t[2])  # por nº de params
    if n_archs >= len(archs):
        return archs
    picks = np.unique(np.linspace(0, len(archs) - 1, n_archs).round().astype(int))
    return [archs[i] for i in picks]


def parse_args():
    p = argparse.ArgumentParser(description="D6 — medição de energia sobre a fronteira de Pareto.")
    p.add_argument("--pareto-json", default=str(ROOT / "analysis" / "nsga2_results.json"))
    p.add_argument("--pareto-key", default="pareto_X_seed42")
    p.add_argument("--n-archs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=4000, help="-1 ou --full p/ treino completo")
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--full", action="store_true", help="usa o dataset completo (mais representativo)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--root", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "results" / "d6_energy.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    train_size = None if (args.full or args.train_size < 0) else args.train_size
    val_size = None if (args.full or args.val_size < 0) else args.val_size

    pareto = json.load(open(args.pareto_json, encoding="utf-8"))[args.pareto_key]
    selected = select_architectures(pareto, args.n_archs)
    print(f"[D6] device={device} | {len(selected)} arquiteturas | epochs={args.epochs} "
          f"| train_size={'full' if train_size is None else train_size}")

    train_loader, val_loader = make_loaders(args.root, args.batch_size, train_size, val_size, SEED)

    rows = []
    t0 = time.time()
    for i, (x, cfg, params) in enumerate(selected, 1):
        mflops = count_mflops(cfg)
        val_acc, kwh, co2 = train_and_measure(cfg, train_loader, val_loader, args.epochs, device)
        row = {
            "arch": genotype_str(x),
            "params": params,
            "mflops": round(mflops, 2),
            "val_acc": round(val_acc, 4),
            "energy_kwh": kwh,
            "co2_kg": co2,
            "energy_wh": round(kwh * 1000, 4),
            "co2_g": round(co2 * 1000, 4),
        }
        rows.append(row)
        print(f"  [{i}/{len(selected)}] params={params:>7,} | {mflops:6.1f} MFLOPs "
              f"| acc={val_acc*100:5.2f}% | {row['energy_wh']:.3f} Wh | {row['co2_g']:.3f} gCO2eq")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[ok] {len(rows)} medições em {time.time()-t0:.1f}s -> {out}")
    print("[nota] energia do treino-proxy; CO2eq depende da intensidade de carbono local (CodeCarbon).")


if __name__ == "__main__":
    main()
