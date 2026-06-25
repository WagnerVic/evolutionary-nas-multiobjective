# Execução via Docker

## Pré-requisitos

- Docker e Docker Compose instalados
- Para GPU: NVIDIA Container Toolkit (`nvidia-docker`)

## Build da imagem

```bash
docker compose build
```

## Download dos dados

```bash
docker compose run --rm nas python scripts/download_data.py
```

## Execução em CPU

### Simulated Annealing

```bash
docker compose run --rm nas python -m experiments.run_sa --device cpu
```

### Baseline (Random Search + LeNet-5)

```bash
docker compose run --rm nas python -m experiments.run_baseline --device cpu
```

## Execução com GPU NVIDIA

### Simulated Annealing

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm nas \
    python -m experiments.run_sa --device cuda
```

### Baseline (Random Search + LeNet-5)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm nas \
    python -m experiments.run_baseline --device cuda
```

## Opções úteis

```bash
# Rodar com subset de sementes (ex: só seed 0 e 1)
... python -m experiments.run_sa --device cuda --seeds 0 1

# Reduzir avaliações para teste rápido
... python -m experiments.run_baseline --device cuda --seeds 0 --n-evals 5

# Alterar épocas proxy / final
... python -m experiments.run_sa --device cuda --epochs-proxy 4 --epochs-final 15
```

## Volumes montados

Os seguintes diretórios são compartilhados entre o host e o container:

| Host | Container | Conteúdo |
|------|-----------|----------|
| `./data/` | `/app/data/` | Datasets (Fashion-MNIST, MNIST) |
| `./results/` | `/app/results/` | CSVs de resultados |
| `./figures/` | `/app/figures/` | Figuras geradas |
| `./analysis/` | `/app/analysis/` | Scripts de análise |

Os resultados aparecem diretamente na pasta local após a execução.

## Verificar GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm nas \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
