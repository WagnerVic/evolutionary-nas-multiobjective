"""Construção da CNN a partir do genótipo e cálculo de f2/f3 (seção 3.2).

`g` deve ser um genótipo já reparado (`genotype.repair`) em todas as
funções deste módulo — elas assumem que os kernels sempre cabem no mapa
espacial corrente (sem essa garantia, `Conv2d` lançaria erro de shape).
"""

import torch.nn as nn

from src.genotype import (
    INPUT_CHANNELS,
    NUM_CLASSES,
    Genotype,
    block_spatial_sizes,
)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"ativação desconhecida: {name}")


class CNNFromGenotype(nn.Module):
    def __init__(self, g: Genotype):
        super().__init__()
        sizes = block_spatial_sizes(g)
        blocks: list[nn.Module] = []
        in_ch = INPUT_CHANNELS
        for i in range(g.l):
            out_ch = g.filters[i]
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=g.kernels[i]))
            blocks.append(_activation(g.acts[i]))
            if g.pools[i]:
                blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        final_size = sizes[-1][2]
        flatten_dim = final_size * final_size * in_ch
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, g.dense_units),
            nn.ReLU(),
            nn.Dropout(g.dropout),
            nn.Linear(g.dense_units, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(g: Genotype) -> nn.Module:
    return CNNFromGenotype(g)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(g: Genotype) -> int:
    """MACs (multiply-accumulates) analítico — proxy determinístico de f3.

    FLOPs ≈ 2×MACs se necessário comparar com trabalhos que reportam FLOPs.
    """
    sizes = block_spatial_sizes(g)
    macs = 0
    in_ch = INPUT_CHANNELS
    for i, (_, conv_size, _) in enumerate(sizes):
        out_ch = g.filters[i]
        macs += (conv_size**2) * out_ch * in_ch * (g.kernels[i] ** 2)
        in_ch = out_ch

    final_size = sizes[-1][2]
    flatten_dim = final_size * final_size * in_ch
    macs += flatten_dim * g.dense_units
    macs += g.dense_units * NUM_CLASSES
    return macs
