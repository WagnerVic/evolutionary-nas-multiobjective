"""Espaço de busca, genótipo e reparo (proposta P8, seções 3.1 e 3.3).

O genótipo é um vetor de tamanho fixo com `MAX_BLOCKS` blocos
convolucionais possíveis; o gene `l` diz quantos estão "ligados" na
arquitetura construída. Os blocos além de `l` continuam no cromossomo
(não são apagados) para que crossover/mutação possam reexpressá-los depois
sem perder informação genética — técnica padrão para profundidade variável
em representações de tamanho fixo.

Todas as variáveis são modeladas como `Choice` (categóricas) para o pymoo,
porque o espaço descrito na proposta é discreto em todos os genes. Isso faz
os operadores default do pymoo para `Choice` (crossover uniforme + mutação
por reamostragem) coincidirem exatamente com o que a seção 3.4 pede.
"""

from dataclasses import dataclass, replace

from pymoo.core.variable import Choice

INPUT_SIZE = 28
INPUT_CHANNELS = 1
NUM_CLASSES = 10

MAX_BLOCKS = 4
L_CHOICES = (2, 3, 4)
MIN_BLOCKS = min(L_CHOICES)

FILTERS = (8, 16, 32, 64)
KERNELS = (3, 5, 7)
POOL = (0, 1)
ACTS = ("relu", "leaky_relu")
DENSE_UNITS = (32, 64, 128)
DROPOUTS = (0.0, 0.25, 0.5)
LRS = (1e-2, 1e-3, 1e-4)

PARAM_CEILING = 500_000


@dataclass(frozen=True)
class Genotype:
    l: int
    filters: tuple[int, ...]
    kernels: tuple[int, ...]
    pools: tuple[int, ...]
    acts: tuple[str, ...]
    dense_units: int
    dropout: float
    lr: float


def pymoo_vars() -> dict[str, Choice]:
    """Declara as variáveis mistas do cromossomo para `Problem(vars=...)`."""
    variables: dict[str, Choice] = {"L": Choice(options=L_CHOICES)}
    for i in range(MAX_BLOCKS):
        variables[f"f{i}"] = Choice(options=FILTERS)
        variables[f"k{i}"] = Choice(options=KERNELS)
        variables[f"p{i}"] = Choice(options=POOL)
        variables[f"a{i}"] = Choice(options=ACTS)
    variables["d"] = Choice(options=DENSE_UNITS)
    variables["dropout"] = Choice(options=DROPOUTS)
    variables["lr"] = Choice(options=LRS)
    return variables


def decode(x: dict) -> Genotype:
    """Converte o dict de variáveis mistas do pymoo em um `Genotype`."""
    return Genotype(
        l=int(x["L"]),
        filters=tuple(int(x[f"f{i}"]) for i in range(MAX_BLOCKS)),
        kernels=tuple(int(x[f"k{i}"]) for i in range(MAX_BLOCKS)),
        pools=tuple(int(x[f"p{i}"]) for i in range(MAX_BLOCKS)),
        acts=tuple(str(x[f"a{i}"]) for i in range(MAX_BLOCKS)),
        dense_units=int(x["d"]),
        dropout=float(x["dropout"]),
        lr=float(x["lr"]),
    )


def _simulate(l: int, kernels: tuple[int, ...], pools: tuple[int, ...]) -> tuple[int, list[int], list[int]]:
    """Simula o encadeamento conv (sem padding) + pooling bloco a bloco.

    Retorna (profundidade_efetiva, kernels_usados, pools_usados). Reduz o
    kernel de um bloco se ele não couber no mapa espacial atual; se nem o
    menor kernel permitido couber, para ali (profundidade efetiva < l).
    """
    size = INPUT_SIZE
    used_kernels = list(kernels)
    used_pools = list(pools)
    effective_l = 0
    for i in range(l):
        feasible = [k for k in KERNELS if k <= size]
        if not feasible:
            break
        k = used_kernels[i] if used_kernels[i] in feasible else max(feasible)
        used_kernels[i] = k
        size = size - k + 1
        if used_pools[i] and size >= 2:
            size //= 2
        else:
            used_pools[i] = 0
        effective_l = i + 1
    return effective_l, used_kernels, used_pools


def repair(g: Genotype) -> Genotype:
    """Repara um genótipo para que a arquitetura resultante seja sempre válida.

    Estratégia (seção 3.3): se o pooling encolher o mapa espacial rápido
    demais e a profundidade efetiva cair abaixo de `l`, tenta de novo sem
    pooling (com kernels <=7 e <=4 blocos, o mapa nunca zera sem pooling:
    28→22→16→10→4), o que garante 2≤l_efetivo≤4 sempre.
    """
    effective_l, kernels, pools = _simulate(g.l, g.kernels, g.pools)
    if effective_l < g.l:
        effective_l, kernels, pools = _simulate(g.l, g.kernels, (0,) * MAX_BLOCKS)
    return replace(g, l=effective_l, kernels=tuple(kernels), pools=tuple(pools))


def block_spatial_sizes(g: Genotype) -> list[tuple[int, int, int]]:
    """Retorna (tamanho_entrada, tamanho_pos_conv, tamanho_pos_pool) por bloco efetivo.

    Assume que `g` já passou por `repair` (kernels sempre cabem no mapa
    espacial corrente).
    """
    sizes = []
    size = INPUT_SIZE
    for i in range(g.l):
        in_size = size
        conv_size = in_size - g.kernels[i] + 1
        out_size = conv_size // 2 if g.pools[i] else conv_size
        sizes.append((in_size, conv_size, out_size))
        size = out_size
    return sizes


def final_spatial_size(g: Genotype) -> int:
    return block_spatial_sizes(g)[-1][2]
