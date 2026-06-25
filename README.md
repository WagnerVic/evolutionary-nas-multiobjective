# NAS leve multiobjetivo

Busca evolutiva de arquiteturas convolucionais (CNNs) compactas para classificação de imagens.

Projeto final da disciplina **INF0415 — Heurísticas e Modelagem Multiobjetivo** (Bacharelado em IA · Instituto de Informática · UFG · 2026/2) · Tema **P8 — Neural Architecture Search leve**.

## Visão geral

Dada uma tarefa de classificação de imagens simples, buscamos **automaticamente** a arquitetura de uma CNN pequena que otimize, **simultaneamente**, três objetivos conflitantes:

- **f₁** — erro de classificação (`1 − acurácia` na validação);
- **f₂** — número de parâmetros treináveis (proxy de tamanho/memória);
- **f₃** — custo de inferência em FLOPs/MACs (proxy determinístico, independente de hardware).

Como não há solução única ótima, o resultado é uma **fronteira de Pareto** de compromissos. Comparamos duas metaheurísticas de paradigmas distintos:

- **Simulated Annealing (SA)** — busca local single-objective via soma ponderada dos objetivos;
- **NSGA-II** (via `pymoo`) — algoritmo evolutivo populacional multiobjetivo.

Como **baseline**: busca aleatória no mesmo espaço de busca + arquitetura fixa LeNet-5.

**Benchmark:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (10 classes, 28×28, 70k imagens). O MNIST é usado como instância secundária para checagem de sanidade.

## Reprodução

Requer Python 3.10+ (testado em 3.12).

```bash
# 1. ambiente e dependências
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. baixar os datasets para data/
python scripts/download_data.py
```

O download usa o `torchvision` (fonte oficial), então não exige conta no Kaggle nem download manual. Opções:

```bash
python scripts/download_data.py --only fashion   # só o Fashion-MNIST
python scripts/download_data.py --only mnist      # só o MNIST (sanity check)
python scripts/download_data.py --root data       # pasta de destino (default: data)
```

Confira o resultado:

```bash
ls data/        # deve conter FashionMNIST/ e MNIST/
```

## Estrutura do repositório

```
evolutionary-nas-multiobjective/
├── README.md
├── requirements.txt
├── scripts/        # utilitários (download de dados, etc.)
├── src/            # código-fonte (genótipo, modelo, avaliação, SA, NSGA-II, baseline)
├── experiments/    # scripts das rodadas (sementes fixadas)
├── results/        # métricas brutas (CSV)
├── figures/        # figuras geradas
├── analysis/       # estatística + geração de figuras
└── data/           # datasets (não versionado)
```

## Docker / SSH

O projeto pode ser executado em container para rodar em outra máquina via SSH.
Veja os comandos em [DOCKER.md](DOCKER.md), incluindo build, download dos dados,
execução em CPU e execução com GPU NVIDIA.

## Equipe

- Raphael Alves de Lima Soares (202403922)
- Wagner Victor Alves de Menezes (202403929)
- Victor Gabriel Ribeiro Jácome (202403926)
- Igor Dias Aguiar (202403907)
- Lucas Fabricio Ozorio (202403914)
