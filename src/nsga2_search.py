"""NSGA-II (pymoo) para o NAS leve multiobjetivo (Checkpoint 2, seção 3.4).

Cada gene do genótipo é modelado como `Choice` (`genotype.pymoo_vars`), o
que faz os operadores *default* do pymoo para variáveis mistas
(`pymoo.core.mixed.MixedVariableMating`) coincidirem exatamente com o que a
proposta pede: `UX` = crossover uniforme, `ChoiceRandomMutation` = mutação
por reamostragem. Não há necessidade de operadores customizados.
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from src.genotype import MAX_BLOCKS, PARAM_CEILING, decode, pymoo_vars
from src.train_eval import evaluate


class NASProblem(ElementwiseProblem):
    """f1 = erro de validação, f2 = nº params, f3 = MACs; G = params - teto (3.3).

    Mantém `self.history`, um log próprio (independente do pymoo) com todo
    indivíduo avaliado — insumo para `history.csv` e para comparações
    futuras de hipervolume com SA/baseline.
    """

    def __init__(self, train_loader, val_loader, epochs, device, seed, param_ceiling=PARAM_CEILING):
        super().__init__(vars=pymoo_vars(), n_obj=3, n_ieq_constr=1)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.param_ceiling = param_ceiling
        self.history: list[dict] = []

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = decode(x)
        result = evaluate(genotype, self.train_loader, self.val_loader, self.epochs, self.device, self.seed)

        out["F"] = [result["f1"], result["f2"], result["f3"]]
        out["G"] = [result["params"] - self.param_ceiling]

        g = result["genotype"]
        row = {
            "eval": len(self.history),
            "l": g.l,
            **{f"f{i}": g.filters[i] for i in range(MAX_BLOCKS)},
            **{f"k{i}": g.kernels[i] for i in range(MAX_BLOCKS)},
            **{f"p{i}": g.pools[i] for i in range(MAX_BLOCKS)},
            **{f"a{i}": g.acts[i] for i in range(MAX_BLOCKS)},
            "dense_units": g.dense_units,
            "dropout": g.dropout,
            "lr": g.lr,
            "val_error": result["f1"],
            "val_acc": result["val_acc"],
            "params": result["params"],
            "macs": result["macs"],
            "feasible": result["params"] <= self.param_ceiling,
        }
        self.history.append(row)


def build_algorithm(pop_size: int) -> NSGA2:
    return NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )


def run(problem: NASProblem, pop_size: int, n_gen: int, seed: int):
    algorithm = build_algorithm(pop_size)
    return minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
