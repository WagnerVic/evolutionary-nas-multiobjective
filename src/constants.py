"""Constantes compartilhadas entre todos os scripts de experimento.

Centralizar aqui evita inconsistências silenciosas entre run_sa.py e
run_baseline.py (ex.: split diferente, warm-up diferente).
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Sementes de controle
# ──────────────────────────────────────────────────────────────────────────────
SPLIT_SEED = 42       # semente do split 50k/10k/10k — idêntica nos dois scripts
N_WARMUP = 30         # arquiteturas amostradas para calcular f2_ref e f3_ref
WARMUP_SEED = 0       # semente do warm-up — garante f2_ref/f3_ref idênticos

# ──────────────────────────────────────────────────────────────────────────────
# Fallbacks de normalização (usados se warm-up falhar)
# ──────────────────────────────────────────────────────────────────────────────
F2_REF_FALLBACK = 500_000       # parâmetros máx razoável
F3_REF_FALLBACK = 50_000_000    # MACs máx razoável

# ──────────────────────────────────────────────────────────────────────────────
# LeNet-5 aproximada como genótipo válido no espaço de busca
# 2 blocos conv, filtros 8/16, kernel 5x5, pooling, ReLU, densa 64, lr 1e-3
# ──────────────────────────────────────────────────────────────────────────────
LENET5_GENOTYPE = np.array([
    0,          # L=2 (índice 0)
    0, 1, 0, 0, # filters: 8, 16, ignorados, ignorados
    1, 1, 0, 0, # kernel:  5,  5, ignorados, ignorados
    1, 1, 0, 0, # pooling: sim, sim, ignorados, ignorados
    0, 0, 0, 0, # activation: ReLU, ReLU, ignorados, ignorados
    1,          # dense: 64
    0,          # dropout: 0.0
    1,          # lr: 1e-3
], dtype=np.int8)

# ──────────────────────────────────────────────────────────────────────────────
# Vetores de peso para o SA (5 pontos no simplex w1+w2+w3=1)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHT_VECTORS = [
    (1.0, 0.0, 0.0),    # só acurácia
    (0.6, 0.2, 0.2),    # acurácia dominante
    (0.34, 0.33, 0.33),  # balanceado
    (0.2, 0.6, 0.2),    # parâmetros dominante
    (0.2, 0.2, 0.6),    # FLOPs dominante
]
