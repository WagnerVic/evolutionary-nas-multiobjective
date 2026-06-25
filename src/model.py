"""Construtor de CNN a partir de um genótipo decodificado.

Estrutura de cada bloco convolucional:
    Conv2d → BatchNorm2d → Activation → [MaxPool2d(2,2) se pooling=1]

Após os L blocos:
    AdaptiveAvgPool2d(1) → Flatten → Linear(dense) → Dropout → Linear(10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchinfo import summary


class _ConvBlock(nn.Module):
    """Um bloco convolucional parametrizado."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_pooling: bool,
        activation: str,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # same padding
            ),
            nn.BatchNorm2d(out_channels),
        ]

        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        if use_pooling:
            layers.append(nn.MaxPool2d(2, 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NASModel(nn.Module):
    """CNN parametrizada pelo genótipo decodificado."""

    def __init__(self, genotype_dict: dict):
        super().__init__()
        blocks_cfg = genotype_dict["blocks"]
        dense_units = genotype_dict["dense_units"]
        dropout_rate = genotype_dict["dropout"]

        conv_blocks: list[nn.Module] = []
        in_ch = 1  # Fashion-MNIST é grayscale

        for blk in blocks_cfg:
            conv_blocks.append(
                _ConvBlock(
                    in_channels=in_ch,
                    out_channels=blk["filters"],
                    kernel_size=blk["kernel"],
                    use_pooling=bool(blk["pooling"]),
                    activation=blk["activation"],
                )
            )
            in_ch = blk["filters"]

        self.features = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(genotype_dict: dict) -> NASModel:
    """Constrói e retorna o modelo a partir de um genótipo decodificado."""
    return NASModel(genotype_dict)


def count_params(model: nn.Module) -> int:
    """Retorna o número total de parâmetros treináveis (f2)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module) -> float:
    """Retorna o total de multiply-accumulate operations (f3) via torchinfo."""
    stats = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    return float(stats.total_mult_adds)
