"""Representação genotípica para o NAS leve multiobjetivo.

Cada indivíduo é um vetor de 20 genes (índices inteiros) de comprimento fixo.
Genes de blocos além de L são mantidos no vetor mas ignorados no decode
(memória genética — ver documentação do plano).
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Espaço de busca — domínios de cada gene
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS: dict[str, list] = {
    "L": [2, 3, 4],
    "filters": [8, 16, 32, 64],
    "kernel": [3, 5, 7],
    "pooling": [0, 1],
    "activation": ["relu", "leakyrelu"],
    "dense": [32, 64, 128],
    "dropout": [0.0, 0.25, 0.5],
    "lr": [1e-2, 1e-3, 1e-4],
}

N_BLOCKS_MAX = 4
N_GENES = 20

# Mapa de posições no vetor
_GENE_RANGES: list[tuple[str, int, int]] = [
    # (nome do domínio, idx_início, idx_fim_exclusivo)
    ("L", 0, 1),
    ("filters", 1, 5),
    ("kernel", 5, 9),
    ("pooling", 9, 13),
    ("activation", 13, 17),
    ("dense", 17, 18),
    ("dropout", 18, 19),
    ("lr", 19, 20),
]


def _domain_size(gene_idx: int) -> int:
    """Retorna o tamanho do domínio para um dado índice de gene."""
    for domain_name, start, end in _GENE_RANGES:
        if start <= gene_idx < end:
            return len(DOMAINS[domain_name])
    raise ValueError(f"gene_idx={gene_idx} fora do intervalo 0..{N_GENES - 1}")


# ──────────────────────────────────────────────────────────────────────────────
# Funções públicas
# ──────────────────────────────────────────────────────────────────────────────


def random_genotype(rng: np.random.Generator) -> np.ndarray:
    """Gera um genótipo aleatório válido."""
    g = np.zeros(N_GENES, dtype=np.int8)
    for i in range(N_GENES):
        g[i] = rng.integers(0, _domain_size(i))
    return g


def decode(g: np.ndarray) -> dict:
    """Converte vetor de índices em dicionário de hiperparâmetros reais.

    Genes de blocos i >= L são omitidos do resultado.
    """
    n_blocks = DOMAINS["L"][g[0]]

    blocks = []
    for i in range(n_blocks):
        blocks.append({
            "filters": DOMAINS["filters"][g[1 + i]],
            "kernel": DOMAINS["kernel"][g[5 + i]],
            "pooling": DOMAINS["pooling"][g[9 + i]],
            "activation": DOMAINS["activation"][g[13 + i]],
        })

    return {
        "n_blocks": n_blocks,
        "blocks": blocks,
        "dense_units": DOMAINS["dense"][g[17]],
        "dropout": DOMAINS["dropout"][g[18]],
        "lr": DOMAINS["lr"][g[19]],
    }


def mutate(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturba exatamente 1 gene aleatório, produzindo valor diferente do atual.

    Decisão de design: quando L muda, os genes dos blocos que se tornam ativos
    mantêm seus últimos valores (memória genética). Não são randomizados.
    """
    g_new = g.copy()
    gene_idx = rng.integers(0, N_GENES)
    dom_size = _domain_size(gene_idx)

    if dom_size <= 1:
        return g_new

    current = g_new[gene_idx]
    choices = [v for v in range(dom_size) if v != current]
    g_new[gene_idx] = rng.choice(choices)
    return g_new


def is_valid(g: np.ndarray) -> bool:
    """Verifica se o genótipo resulta em mapa espacial >= 1x1.

    Com same-padding (padding=kernel//2) a dimensão só diminui via MaxPool(2,2).
    Entrada: 28x28. Cada pooling divide por 2.
    """
    n_blocks = DOMAINS["L"][g[0]]
    spatial = 28
    for i in range(n_blocks):
        if DOMAINS["pooling"][g[9 + i]] == 1:
            spatial = spatial // 2
    # AdaptiveAvgPool2d(1) cuida do resto, mas verificamos por segurança
    return spatial >= 1
