"""Geração de figuras e análise estatística: SA vs Random Search vs LeNet-5.

Uso:
    python -m analysis.generate_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

WEIGHT_LABELS = {
    (1.0, 0.0, 0.0): "Acc only\n(1,0,0)",
    (0.6, 0.2, 0.2): "Acc dom.\n(0.6,0.2,0.2)",
    (0.34, 0.33, 0.33): "Balanced\n(⅓,⅓,⅓)",
    (0.2, 0.6, 0.2): "Params dom.\n(0.2,0.6,0.2)",
    (0.2, 0.2, 0.6): "FLOPs dom.\n(0.2,0.2,0.6)",
}


def _weight_key(row) -> tuple:
    return (round(row["w1"], 2), round(row["w2"], 2), round(row["w3"], 2))


# ──────────────────────────────────────────────────────────────────────────────
# Carregar dados
# ──────────────────────────────────────────────────────────────────────────────

sa_raw = pd.read_csv(RESULTS_DIR / "sa_raw.csv")
sa_history = pd.read_csv(RESULTS_DIR / "sa_history.csv")
sa_test = pd.read_csv(RESULTS_DIR / "sa_test_eval.csv")

baseline_raw = pd.read_csv(RESULTS_DIR / "baseline_raw.csv")
baseline_test = pd.read_csv(RESULTS_DIR / "baseline_test_eval.csv")

rs_raw = baseline_raw[baseline_raw["method"] == "random_search"].copy()
lenet_raw = baseline_raw[baseline_raw["method"] == "lenet5_approx"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Figura 1: Curva de convergência do SA (fitness × iteração)
# ──────────────────────────────────────────────────────────────────────────────

def fig_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))

    for (w1, w2, w3), grp in sa_history.groupby(["w1", "w2", "w3"]):
        label = WEIGHT_LABELS.get((round(w1, 2), round(w2, 2), round(w3, 2)), f"({w1},{w2},{w3})")
        pivot = grp.pivot(index="iter", columns="seed", values="fitness")
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)
        ax.plot(mean.index, mean.values, label=label)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Iteração")
    ax.set_ylabel("Fitness (soma ponderada normalizada)")
    ax.set_title("Convergência do Simulated Annealing — média ± desvio (10 seeds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig1_convergence_sa.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig1_convergence_sa.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 2: Boxplot comparativo SA vs Random Search (f1 = erro na validação)
# ──────────────────────────────────────────────────────────────────────────────

def fig_boxplot_f1_validation():
    sa_plot = sa_raw.copy()
    sa_plot["method"] = "SA"
    rs_plot = rs_raw.copy()
    rs_plot["method"] = "Random Search"

    combined = pd.concat([sa_plot, rs_plot], ignore_index=True)
    combined["weight_label"] = combined.apply(
        lambda r: WEIGHT_LABELS.get(_weight_key(r), ""), axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=combined, x="weight_label", y="f1", hue="method",
        ax=ax, palette=["#2196F3", "#FF9800"],
    )
    ax.set_xlabel("Vetor de pesos")
    ax.set_ylabel("f₁ (1 − acurácia validação)")
    ax.set_title("SA vs Random Search — Erro na validação por vetor de pesos (10 seeds)")
    ax.legend(title="Método")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(FIGURES_DIR / "fig2_boxplot_f1_validation.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig2_boxplot_f1_validation.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 3: Boxplot de acurácia no teste (SA vs Random Search vs LeNet-5)
# ──────────────────────────────────────────────────────────────────────────────

def fig_boxplot_test_acc():
    sa_t = sa_test.copy()
    sa_t["method"] = "SA"

    rs_t = baseline_test.copy()
    rs_t["method"] = "Random Search"

    lenet_t = lenet_raw.copy()
    lenet_t["test_acc"] = 1.0 - lenet_t["f1"]
    lenet_t["method"] = "LeNet-5"
    lenet_t["w1"] = 1.0
    lenet_t["w2"] = 0.0
    lenet_t["w3"] = 0.0

    combined = pd.concat([
        sa_t[["method", "seed", "w1", "w2", "w3", "test_acc"]],
        rs_t[["method", "seed", "w1", "w2", "w3", "test_acc"]],
        lenet_t[["method", "seed", "w1", "w2", "w3", "test_acc"]],
    ], ignore_index=True)

    combined["weight_label"] = combined.apply(
        lambda r: WEIGHT_LABELS.get(_weight_key(r), "LeNet-5\n(fixo)"), axis=1
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=combined, x="weight_label", y="test_acc", hue="method",
        ax=ax, palette=["#2196F3", "#FF9800", "#4CAF50"],
    )
    ax.set_xlabel("Vetor de pesos")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title("Acurácia no teste — SA vs Random Search vs LeNet-5 (15 épocas, 10 seeds)")
    ax.legend(title="Método")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.70, 0.95)
    fig.savefig(FIGURES_DIR / "fig3_boxplot_test_acc.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig3_boxplot_test_acc.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 4: Scatter f1 × f2 (erro × parâmetros) — validação
# ──────────────────────────────────────────────────────────────────────────────

def fig_scatter_f1_f2():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(sa_raw["f2"], sa_raw["f1"], label="SA", alpha=0.7, marker="o", s=40)
    ax.scatter(rs_raw["f2"], rs_raw["f1"], label="Random Search", alpha=0.7, marker="^", s=40)
    # LeNet-5 omitida: seu f1 disponível é de TESTE, não de validação — plotá-la aqui
    # misturaria eixos não comparáveis (ver relatorio_erros_SA.md, erro #5). A comparação
    # com a LeNet-5 aparece na Fig. 3 (boxplot de acurácia no teste).

    ax.set_xlabel("f₂ (parâmetros treináveis)")
    ax.set_ylabel("f₁ (1 − acurácia validação)")
    ax.set_title("Trade-off: Erro × Parâmetros (validação) — SA vs Random Search")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig4_scatter_f1_f2.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig4_scatter_f1_f2.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Figura 5: Scatter f1 × f3 (erro × MACs) — validação
# ──────────────────────────────────────────────────────────────────────────────

def fig_scatter_f1_f3():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(sa_raw["f3"], sa_raw["f1"], label="SA", alpha=0.7, marker="o", s=40)
    ax.scatter(rs_raw["f3"], rs_raw["f1"], label="Random Search", alpha=0.7, marker="^", s=40)
    # LeNet-5 omitida: seu f1 disponível é de TESTE, não de validação (ver erro #5).
    # A comparação com a LeNet-5 aparece na Fig. 3 (boxplot de acurácia no teste).

    ax.set_xlabel("f₃ (MACs — custo de inferência)")
    ax.set_ylabel("f₁ (1 − acurácia validação)")
    ax.set_title("Trade-off: Erro × MACs (validação) — SA vs Random Search")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "fig5_scatter_f1_f3.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig5_scatter_f1_f3.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Análise estatística: Wilcoxon pareado + rank-biserial + Bonferroni
# ──────────────────────────────────────────────────────────────────────────────

def rank_biserial(x, y):
    """Calcula rank-biserial correlation como tamanho do efeito para Wilcoxon."""
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    r_plus = np.sum(ranks[diff > 0])
    r_minus = np.sum(ranks[diff < 0])
    return (r_plus - r_minus) / (r_plus + r_minus)


def statistical_analysis():
    """Wilcoxon pareado (SA vs Random Search) por vetor de peso."""
    weight_vectors = [(1.0, 0.0, 0.0), (0.6, 0.2, 0.2), (0.34, 0.33, 0.33),
                      (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)]
    n_comparisons = len(weight_vectors)
    alpha = 0.05
    alpha_bonf = alpha / n_comparisons

    results_rows = []

    print("\n" + "=" * 80)
    print("ANÁLISE ESTATÍSTICA: SA vs Random Search (Wilcoxon pareado)")
    print(f"Correção de Bonferroni: α={alpha} / {n_comparisons} = {alpha_bonf:.4f}")
    print("=" * 80)

    for w1, w2, w3 in weight_vectors:
        # Pareamento correto por SEMENTE: cada seed contribui com 1 par (SA, RS).
        # NÃO ordenar os vetores — np.sort destruiria o pareamento por seed e o teste
        # deixaria de ser um Wilcoxon pareado válido (ver relatorio_erros_SA.md, erro #1).
        sa_w = sa_raw[(sa_raw["w1"] == w1) & (sa_raw["w2"] == w2) & (sa_raw["w3"] == w3)][["seed", "f1"]]
        rs_w = rs_raw[(rs_raw["w1"] == w1) & (rs_raw["w2"] == w2) & (rs_raw["w3"] == w3)][["seed", "f1"]]

        paired = sa_w.merge(rs_w, on="seed", suffixes=("_sa", "_rs")).sort_values("seed")
        sa_f1 = paired["f1_sa"].values
        rs_f1 = paired["f1_rs"].values

        # Wilcoxon signed-rank test (pareado por seed)
        stat_w, p_value = stats.wilcoxon(sa_f1, rs_f1, alternative="two-sided")
        r_effect = rank_biserial(sa_f1, rs_f1)
        significant = p_value < alpha_bonf

        label = WEIGHT_LABELS[(w1, w2, w3)].replace("\n", " ")
        print(f"\n  Peso: {label}")
        print(f"    SA   f1: média={sa_f1.mean():.4f}, std={sa_f1.std():.4f}")
        print(f"    RS   f1: média={rs_f1.mean():.4f}, std={rs_f1.std():.4f}")
        print(f"    Wilcoxon W={stat_w:.1f}, p={p_value:.6f}")
        print(f"    Rank-biserial r={r_effect:.4f}")
        print(f"    Significativo (p < {alpha_bonf:.4f})? {'SIM' if significant else 'NÃO'}")

        results_rows.append({
            "weight": f"({w1},{w2},{w3})",
            "sa_mean_f1": sa_f1.mean(),
            "sa_std_f1": sa_f1.std(),
            "rs_mean_f1": rs_f1.mean(),
            "rs_std_f1": rs_f1.std(),
            "wilcoxon_W": stat_w,
            "p_value": p_value,
            "p_bonferroni": min(p_value * n_comparisons, 1.0),
            "rank_biserial_r": r_effect,
            "significant": significant,
        })

    print("\n" + "=" * 80)

    df_stats = pd.DataFrame(results_rows)
    df_stats.to_csv(RESULTS_DIR / "statistical_analysis.csv", index=False)
    print(f"\n[salvo] {RESULTS_DIR / 'statistical_analysis.csv'}")

    return df_stats


# ──────────────────────────────────────────────────────────────────────────────
# Figura 6: Tabela visual da análise estatística
# ──────────────────────────────────────────────────────────────────────────────

def fig_stats_table(df_stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table_data = []
    for _, row in df_stats.iterrows():
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["significant"] else "ns"))
        table_data.append([
            row["weight"],
            f"{row['sa_mean_f1']:.4f} ± {row['sa_std_f1']:.4f}",
            f"{row['rs_mean_f1']:.4f} ± {row['rs_std_f1']:.4f}",
            f"{row['p_value']:.5f}",
            f"{row['rank_biserial_r']:.3f}",
            sig,
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Pesos", "SA (f₁)", "Random Search (f₁)", "p-value", "r (effect)", "Sig."],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Wilcoxon pareado — SA vs Random Search (α=0.01, Bonferroni)", fontsize=11, pad=20)

    fig.savefig(FIGURES_DIR / "fig6_stats_table.png")
    plt.close(fig)
    print(f"[salvo] {FIGURES_DIR / 'fig6_stats_table.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Gerando figuras...\n")

    fig_convergence()
    fig_boxplot_f1_validation()
    fig_boxplot_test_acc()
    fig_scatter_f1_f2()
    fig_scatter_f1_f3()

    print("\nExecutando análise estatística...")
    df_stats = statistical_analysis()
    fig_stats_table(df_stats)

    print(f"\n[concluído] {len(list(FIGURES_DIR.glob('*.png')))} figuras em {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
