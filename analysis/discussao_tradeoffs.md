# Discussão de Trade-offs — NAS Multiobjetivo (Checkpoint 2)
**INF0415 · Heurísticas e Modelagem · UFG 2026/2**  
Equipe: Raphael Alves · Wagner Victor · Victor Gabriel · Igor Dias · Lucas Fabricio

---

## 1. Contexto e formulação

O problema de Neural Architecture Search (NAS) leve foi formulado como um problema de otimização multiobjetivo com três objetivos conflitantes a minimizar simultaneamente:

- **f₁ = 1 − acurácia de validação** — qualidade preditiva
- **f₂ = número de parâmetros treináveis** — proxy de memória e tamanho do modelo
- **f₃ = FLOPs de inferência** — proxy de custo computacional, independente de hardware

O espaço de busca consiste em vetores inteiros de 14 genes descrevendo arquiteturas CNN com 2 a 4 blocos convolucionais, variando filtros, kernels, pooling, ativação, unidades densas e taxa de aprendizado. O benchmark principal é o Fashion-MNIST (10 classes, 28×28, 70k imagens); o MNIST serve como instância secundária de sanidade.

---

## 2. Resultados quantitativos

| Método | Val acc (%) | Params médios | FLOPs médios | HV |
|---|---|---|---|---|
| **NSGA-II** | **93.17 ± 0.39** | — (fronteira) | — (fronteira) | **924.04 ± 3.08** |
| SA (Checkpoint 1) | 90.50 ± 0.60 | 147.2k | 17.0M | — |
| Busca Aleatória | 82.40 ± 9.68 | 62.2k | 48.0M | — |
| LeNet-5 (fixo) | 89.71 | 61.8k | 0.85M | — |

> **Teste Mann-Whitney U: p = 0.000 (significativo).** Cliff's delta indica tamanho de efeito grande — NSGA-II supera consistentemente o SA em acurácia de validação. A comparação é não-pareada (sementes 0–9 no SA vs. 42,7,13,… no NSGA-II), o que é metodologicamente correto para Mann-Whitney.

---

## 3. Trade-offs na fronteira de Pareto

### 3.1 Acurácia × Parâmetros (f₁ × f₂)

A Figura 1 (subplot esquerdo) revela o trade-off mais pronunciado do problema. Arquiteturas com acurácia de validação acima de 92% exigem sistematicamente mais de 100k parâmetros, com dois pontos Pareto chegando a 250k e 120k. Abaixo de 90k parâmetros, a acurácia cai abruptamente — evidência de que existe um limiar arquitetural mínimo para o Fashion-MNIST. A busca aleatória ocupa a região inferior-esquerda do gráfico de forma difusa, enquanto o NSGA-II concentra suas soluções Pareto na região de alto desempenho, indicando que a pressão seletiva do crowding distance efetivamente diversificou a população ao longo da fronteira.

A LeNet-5 aparece como ponto de referência interessante: ~62k parâmetros e ~90% de acurácia — bom custo-benefício para uma arquitetura fixa, mas dominada por várias soluções Pareto do NSGA-II que alcançam acurácia maior com parâmetros similares ou ligeiramente maiores.

### 3.2 Acurácia × FLOPs (f₁ × f₃)

O subplot central mostra um comportamento inesperado: os FLOPs das soluções Pareto variam pouco em função da acurácia. A maioria das soluções Pareto se concentra abaixo de 100 MFLOPs, independentemente da acurácia alcançada. Dois outliers chegam a ~400 MFLOPs. Isso sugere que o NSGA-II convergiu para arquiteturas eficientes em FLOPs como estratégia dominante — kernels 3×3 com pooling são mais frequentes na fronteira (confirmado pela Figura 4), e o `AdaptiveAvgPool2d` elimina FLOPs redundantes antes da camada densa.

A LeNet-5 tem apenas 0.85 MFLOPs — ordem de magnitude abaixo de todos os outros pontos — mas sua acurácia é inferior ao topo da fronteira Pareto. Esse trade-off sugere que kernels 5×5 da LeNet-5 são mais eficientes em FLOPs por parâmetro, mas a arquitetura é limitada pela profundidade e pela ausência de BatchNorm.

### 3.3 Parâmetros × FLOPs (f₂ × f₃)

O subplot direito é o mais revelador sobre a estrutura interna do espaço de busca. A correlação entre parâmetros e FLOPs **não é linear**: alguns pontos têm muitos parâmetros e poucos FLOPs, e vice-versa. Isso ocorre porque:

- **Camadas densas** aumentam parâmetros sem impacto proporcional em FLOPs convolucionais
- **Max-pooling** reduz FLOPs nas camadas subsequentes sem reduzir parâmetros das camadas anteriores
- **Kernels grandes** (5×5, 7×7) elevam FLOPs quadraticamente, mas parâmetros apenas linearmente em relação à área do kernel

Essa desacoplagem f₂/f₃ valida a escolha de tratá-los como dois objetivos distintos: uma formulação single-objective com agregação ponderada não capturaria essa estrutura.

---

## 4. NSGA-II × SA: análise comparativa

### 4.1 Natureza da comparação

O SA (Checkpoint 1) é single-objective por execução: cada rodada usa um vetor de pesos fixo (w₁, w₂, w₃) que agrega os três objetivos em escalar único via soma ponderada. Para mapear a fronteira de Pareto completa, seria necessário repetir a busca com múltiplos vetores de peso — o que foi feito com 5 configurações, totalizando 50 execuções (10 seeds × 5 pesos).

O NSGA-II mantém uma população de soluções não-dominadas em cada geração e produz toda a fronteira em uma única execução. Isso é uma vantagem estrutural em problemas onde os trade-offs entre objetivos são desconhecidos a priori.

### 4.2 Qualidade das soluções

A diferença de 2.67 pontos percentuais em acurácia de validação (93.17% vs 90.50%) com p < 0.001 e tamanho de efeito grande é substancial no contexto de NAS. Entretanto, essa comparação tem um viés importante: **o NSGA-II usa uma GPU RTX 4090 e avalia populações inteiras em paralelo**, enquanto o SA roda sequencialmente. O tempo de parede não é diretamente comparável.

### 4.3 Convergência (Figura 3, subplot direito)

O gráfico de duplo eixo mostra curvas de convergência qualitativas. O NSGA-II (HV, eixo esquerdo) cresce monotonicamente da geração 1 à 15, com banda de variância estreita nas últimas gerações — indicando convergência estável. O SA (best acc acumulada, eixo direito) converge mais rapidamente nas primeiras iterações mas estabiliza abaixo do NSGA-II. Isso é coerente com a natureza de cada algoritmo: o SA explora via Metropolis com resfriamento, tendendo a estabilizar quando a temperatura cai; o NSGA-II continua pressionando a fronteira via dominância de Pareto e crowding distance.

> **Limitação metodológica:** os eixos x do NSGA-II (gerações) e do SA (iterações) têm semânticas diferentes e foram normalizados apenas para comparação visual. A comparação de convergência por número de avaliações de função seria mais rigorosa.

---

## 5. Análise dos genes na fronteira de Pareto (Figura 4)

### 5.1 Heatmap de genes

O heatmap mostra alta variabilidade nos genes de filtros (f1, f2, f3) e de kernel (k1, k2, k3), e baixa variabilidade no gene de pooling p3 e na taxa de aprendizado (lr). Isso sugere que:

- **Filtros e kernels**: principais eixos de diversidade na fronteira. O NSGA-II explorou amplamente esses genes, produzindo soluções com diferentes balanços de capacidade vs. eficiência.
- **Pooling p3** (bloco 3): quase sempre ativo nas soluções Pareto — reduzir a dimensão espacial no terceiro bloco é consistentemente benéfico.
- **Taxa de aprendizado**: concentração em lr=1e-3. Sob orçamento fixo de 5 épocas, lrs menores (1e-4) convergem devagar demais e lrs maiores (1e-2) oscilam — o NSGA-II aprendeu essa preferência indiretamente via seleção.

### 5.2 Distribuição de f1 (filtros do bloco 1)

O barplot mostra forte preferência por **8 filtros no bloco 1** (~105 ocorrências), seguido de 32 e 16, com 64 sendo raro. Isso é contra-intuitivo à primeira vista — poderíamos esperar que mais filtros no primeiro bloco melhorasse a representação.

A explicação mais provável é que o primeiro bloco opera sobre a imagem 28×28 com 1 canal: 8 filtros 3×3 extraem features básicas (bordas, texturas) de forma suficiente, e o custo de aumentar para 64 filtros no bloco 1 é desproporcional em parâmetros e FLOPs. A riqueza de representação cresce nos blocos subsequentes, onde o mapa espacial já foi reduzido pelo pooling.

---

## 6. Erros e limitações identificadas

### 6.1 Problemas metodológicos

**Comparação de acurácia NSGA-II vs. SA não é inteiramente justa.**  
O NSGA-II reporta a melhor acurácia entre todas as soluções da fronteira de Pareto (93.17%), que é naturalmente a solução que maximiza f₁ ignorando f₂ e f₃. O SA reporta a melhor acurácia dentre configurações de peso que incluem termos para f₂ e f₃. Uma comparação mais justa seria: "melhor acurácia do NSGA-II com restrição de parâmetros ≤ X" vs. "SA com w₁=1.0" — que são metodologicamente equivalentes.

**Orçamento de épocas fixo subestima arquiteturas de convergência lenta.**  
Com apenas 5 épocas de treino por avaliação, modelos maiores ou com lr=1e-3/1e-4 não convergem completamente. Isso penaliza injustamente arquiteturas profundas e favorece arquiteturas rasas com lr alto. A avaliação final com 15 épocas atenua parcialmente esse problema, mas a busca em si foi conduzida com o viés de 5 épocas.

**Espaço de busca assimétrico para blocos 3 e 4.**  
Na codificação atual, os blocos 3 e 4 compartilham os genes f3, k3, p3 (índices 9–11), e reutilizam a1/a2 como ativação. Isso significa que arquiteturas com L=4 blocos são menos diversas internamente — os dois últimos blocos têm configurações parcialmente iguais, o que restringe o espaço de busca real para L=4.

### 6.2 Problemas nas visualizações

**Figura 1 — linha de fronteira enganosa.**  
A linha que conecta os pontos Pareto no subplot "Acc × Parâmetros" foi traçada ordenando por acurácia, mas a fronteira de Pareto em 3 objetivos projetada em 2D não necessariamente forma uma curva monotônica nessa projeção. A linha pode induzir interpretação incorreta de que todos os pontos intermediários são Pareto-ótimos nessa projeção — o que não é garantido.

**Figura 3 — eixo duplo com escalas incomparáveis.**  
O HV do NSGA-II (eixo esquerdo, ~890–925) e a val_acc do SA (eixo direito, ~82–92%) são métricas fundamentalmente diferentes. A comparação visual da "velocidade de convergência" é qualitativa, não quantitativa. Isso deve ser explicitado na legenda ou caption do trabalho final.

**Figura 4 — frequência absoluta vs. relativa.**  
O barplot de distribuição de f1 usa frequência absoluta (~100 ocorrências), mas o número total de soluções Pareto varia entre sementes. Frequência relativa (proporção) seria mais informativa para comparar a preferência do algoritmo independentemente do tamanho da fronteira por semente.

### 6.3 Pontos de atenção para o relatório

- O **hipervolume** (924.04) é calculado com ponto de referência `[1.0, 5.0, 200.0]` (f₁, f₂/1e5, f₃/1e7). Esse ponto deve ser reportado explicitamente — valores de HV sem referência são incomparáveis entre trabalhos.
- A **busca aleatória** (82.40% ± 9.68%) tem desvio padrão muito alto (±9.68pp), refletindo a variabilidade intrínseca do espaço. Isso é esperado e fortalece o argumento de que metaheurísticas guiadas superam a busca aleatória de forma consistente.
- Os **resultados do SA** (90.50%) são competitivos com a LeNet-5 (89.71%) e coerentes com o esperado para single-objective NAS com orçamento equivalente.

---

## 7. Conclusões

O NSGA-II demonstrou superioridade estatisticamente significativa sobre o SA em acurácia de validação (93.17% vs. 90.50%, p < 0.001), e sua capacidade de produzir uma fronteira de Pareto diversificada em uma única execução é uma vantagem estrutural para o problema. A análise dos genes revelou padrões interpretáveis: preferência por poucos filtros no primeiro bloco, kernels 3×3, pooling no terceiro bloco e lr=1e-3. A busca aleatória, apesar de usar o mesmo orçamento de avaliações, produziu soluções muito inferiores e com alta variância — confirmando o valor das metaheurísticas guiadas para NAS.

As principais limitações identificadas — orçamento de épocas limitado, comparação de acurácia não completamente simétrica entre NSGA-II e SA, e assimetria no genótipo para L=4 blocos — não invalidam os resultados, mas devem ser reconhecidas na discussão do trabalho final para garantir rigor científico.
