"""
Experiment 3 – CEC 2009 Unconstrained Functions (UF1-UF3) Benchmark
===================================================================
Compares the Improved Multi-Objective Phototropic Growth Algorithm (MOPGA)
against NSGA-II on three 2-objective CEC 2009 UF benchmark problems.

The UF problems are implemented directly since pymoo 0.6.x does not ship
with them in its problem registry.

Metrics evaluated: Hypervolume (HV), Inverted Generational Distance (IGD),
and Spacing (SP).

Usage:
    python experiment3.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Custom algorithm import ---
from mopga_improved import run_mopga

# --- pymoo imports ---
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


# =====================================================================
# CEC 2009 Unconstrained Problems (UF1, UF2, UF3)
# Reference: Q. Zhang et al., "Multiobjective optimization Test Instances
#            for the CEC 2009 Special Session and Competition", 2008
# =====================================================================

class UF1(Problem):
    """
    CEC 2009 UF1

    Search space: x1 ∈ [0,1], xj ∈ [-1,1] for j = 2,...,n
    Pareto front: f2 = 1 - sqrt(f1), 0 ≤ f1 ≤ 1
    """

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0)
        xu = np.full(n_var, 1.0)
        xl[0] = 0.0
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var
        x1 = X[:, 0]

        J1 = np.arange(2, n + 1, 2) - 1  # odd j (0-indexed: 1,3,5,...)
        J2 = np.arange(3, n + 1, 2) - 1  # even j (0-indexed: 2,4,6,...)

        sum1 = np.zeros(len(X))
        for j in J1:
            yj = X[:, j] - np.sin(6.0 * np.pi * x1 + (j + 1) * np.pi / n)
            sum1 += yj ** 2

        sum2 = np.zeros(len(X))
        for j in J2:
            yj = X[:, j] - np.sin(6.0 * np.pi * x1 + (j + 1) * np.pi / n)
            sum2 += yj ** 2

        f1 = x1 + 2.0 / len(J1) * sum1
        f2 = 1.0 - np.sqrt(x1) + 2.0 / len(J2) * sum2

        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class UF2(Problem):
    """
    CEC 2009 UF2

    Search space: x1 ∈ [0,1], xj ∈ [-1,1] for j = 2,...,n
    Pareto front: f2 = 1 - sqrt(f1), 0 ≤ f1 ≤ 1
    """

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0)
        xu = np.full(n_var, 1.0)
        xl[0] = 0.0
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var
        x1 = X[:, 0]

        J1 = np.arange(2, n + 1, 2) - 1  # odd j
        J2 = np.arange(3, n + 1, 2) - 1  # even j

        sum1 = np.zeros(len(X))
        for j in J1:
            jp1 = j + 1  # 1-indexed j
            yj = X[:, j] - (
                0.3 * x1 ** 2 * np.cos(24.0 * np.pi * x1 + 4.0 * jp1 * np.pi / n)
                + 0.6 * x1
            ) * np.sin(6.0 * np.pi * x1 + jp1 * np.pi / n)
            sum1 += yj ** 2

        sum2 = np.zeros(len(X))
        for j in J2:
            jp1 = j + 1
            yj = X[:, j] - (
                0.3 * x1 ** 2 * np.cos(24.0 * np.pi * x1 + 4.0 * jp1 * np.pi / n)
                + 0.6 * x1
            ) * np.cos(6.0 * np.pi * x1 + jp1 * np.pi / n)
            sum2 += yj ** 2

        f1 = x1 + 2.0 / len(J1) * sum1
        f2 = 1.0 - np.sqrt(x1) + 2.0 / len(J2) * sum2

        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class UF3(Problem):
    """
    CEC 2009 UF3

    Search space: xj ∈ [0,1] for j = 1,...,n
    Pareto front: f2 = 1 - sqrt(f1), 0 ≤ f1 ≤ 1
    """

    def __init__(self, n_var=30):
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var
        x1 = X[:, 0]

        J1 = np.arange(2, n + 1, 2) - 1  # odd j
        J2 = np.arange(3, n + 1, 2) - 1  # even j

        sum1 = np.zeros(len(X))
        for j in J1:
            jp1 = j + 1  # 1-indexed j
            exponent = 0.5 * (1.0 + 3.0 * (jp1 - 2.0) / (n - 2.0))
            yj = X[:, j] - np.power(x1, exponent)
            sum1 += yj ** 2

        sum2 = np.zeros(len(X))
        for j in J2:
            jp1 = j + 1
            exponent = 0.5 * (1.0 + 3.0 * (jp1 - 2.0) / (n - 2.0))
            yj = X[:, j] - np.power(x1, exponent)
            sum2 += yj ** 2

        f1 = x1 + 2.0 / len(J1) * sum1
        f2 = 1.0 - np.sqrt(x1) + 2.0 / len(J2) * sum2

        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


# =====================================================================
# Configuration
# =====================================================================

PROBLEMS = {
    "uf1": UF1,
    "uf2": UF2,
    "uf3": UF3,
}
POP_SIZE = 100
GENERATIONS = 250
SEED = 1
HV_REF_POINT = np.array([2.0, 2.0])


# =====================================================================
# Utility functions
# =====================================================================

def compute_spacing(F: np.ndarray) -> float:
    """
    Compute the Spacing indicator for a set of non-dominated solutions.

    Spacing measures how uniformly the solutions are distributed along the
    obtained Pareto front. A lower value indicates a more uniform spread.

    SP = sqrt( (1/n) * sum_i (d_i - d_mean)^2 )

    where d_i is the minimum Euclidean distance from solution i to every
    other solution in the set.
    """
    if F is None or len(F) < 2:
        return 0.0

    n = len(F)
    d = np.full(n, np.inf)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(F[i] - F[j])
            if dist < d[i]:
                d[i] = dist

    d_mean = np.mean(d)
    spacing = np.sqrt(np.mean((d - d_mean) ** 2))
    return spacing


def compute_metrics(F: np.ndarray, pf: np.ndarray) -> dict:
    """Compute HV, IGD, and Spacing for a given front F against the true PF."""
    if F is None or len(F) == 0:
        return {"HV": 0.0, "IGD": float("inf"), "SP": 0.0}

    # Hypervolume (higher is better)
    hv_indicator = HV(ref_point=HV_REF_POINT)
    hv_value = hv_indicator(F)

    # IGD (lower is better)
    igd_indicator = IGD(pf)
    igd_value = igd_indicator(F)

    # Spacing (lower is better → more uniform)
    sp_value = compute_spacing(F)

    return {"HV": float(hv_value), "IGD": float(igd_value), "SP": float(sp_value)}


# =====================================================================
# Runners
# =====================================================================

def run_nsga2(problem) -> np.ndarray:
    """Run pymoo's NSGA-II and return the objective values of the result."""
    algorithm = NSGA2(pop_size=POP_SIZE)
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", GENERATIONS),
        seed=SEED,
        verbose=False,
    )
    return res.F


def run_mopga_wrapper(problem) -> np.ndarray:
    """Run our custom Improved MOPGA and return objective values."""
    return run_mopga(problem, pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED)


# =====================================================================
# Output helpers
# =====================================================================

def print_comparison_table(problem_name: str, nsga2_metrics: dict, mopga_metrics: dict):
    """Print a neatly formatted comparison table for a single problem."""
    width = 62
    print("\n" + "=" * width)
    print(f"  Benchmark: {problem_name.upper()}")
    print("=" * width)
    print(f"  {'Metric':<12} {'NSGA-II':>16}   {'MOPGA Improved':>16}")
    print("-" * width)
    for key in ("HV", "IGD", "SP"):
        nsga2_val = nsga2_metrics[key]
        mopga_val = mopga_metrics[key]

        # Determine which algorithm is better for this metric
        if key == "HV":
            better = "MOPGA" if mopga_val > nsga2_val else "NSGA-II"
        else:
            better = "MOPGA" if mopga_val < nsga2_val else "NSGA-II"

        marker_n = " <" if better == "NSGA-II" else ""
        marker_m = " <" if better == "MOPGA" else ""

        print(
            f"  {key:<12} {nsga2_val:>14.6f}{marker_n:>2}   {mopga_val:>14.6f}{marker_m:>2}"
        )
    print("=" * width)
    print("  < = better\n")


def plot_results(
    problem_name: str,
    pf: np.ndarray,
    nsga2_F: np.ndarray,
    mopga_F: np.ndarray,
):
    """Generate a 2D scatter plot comparing both algorithms against the true PF."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # True Pareto Front
    ax.plot(
        pf[:, 0],
        pf[:, 1],
        color="grey",
        linewidth=1.5,
        alpha=0.5,
        label="True Pareto Front",
        zorder=1,
    )

    # NSGA-II results
    ax.scatter(
        nsga2_F[:, 0],
        nsga2_F[:, 1],
        marker="x",
        color="red",
        s=40,
        alpha=0.8,
        label="NSGA-II",
        zorder=3,
    )

    # MOPGA results
    ax.scatter(
        mopga_F[:, 0],
        mopga_F[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
        s=40,
        alpha=0.8,
        label="MOPGA Improved",
        zorder=2,
    )

    ax.set_xlabel(r"$f_1$", fontsize=13)
    ax.set_ylabel(r"$f_2$", fontsize=13)
    ax.set_title(f"Performance on {problem_name.upper()}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = f"{problem_name}_experiment3.png"
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved -> {save_path}")
    plt.show()


# =====================================================================
# Main experiment loop
# =====================================================================

def run_experiment():
    """Execute the full CEC 2009 UF benchmark experiment."""
    print("=" * 62)
    print("  Experiment 3: CEC 2009 Unconstrained Functions (UF)")
    print(f"  Population: {POP_SIZE}  |  Generations: {GENERATIONS}  |  Seed: {SEED}")
    print("=" * 62)

    for problem_name, ProblemClass in PROBLEMS.items():
        print(f"\n>>> Loading problem: {problem_name.upper()} ...")
        problem = ProblemClass()
        pf = problem.pareto_front()

        # --- Run NSGA-II ---
        print(f"  Running NSGA-II on {problem_name.upper()} ...")
        nsga2_F = run_nsga2(problem)
        print(f"  NSGA-II complete  ({len(nsga2_F)} solutions)")

        # --- Run Improved MOPGA ---
        print(f"  Running MOPGA Improved on {problem_name.upper()} ...")
        mopga_F = run_mopga_wrapper(problem)
        print(f"  MOPGA complete    ({len(mopga_F)} solutions)")

        # --- Metrics ---
        nsga2_metrics = compute_metrics(nsga2_F, pf)
        mopga_metrics = compute_metrics(mopga_F, pf)

        # --- Console output ---
        print_comparison_table(problem_name, nsga2_metrics, mopga_metrics)

        # --- Visualization ---
        plot_results(problem_name, pf, nsga2_F, mopga_F)

    print("\n[OK] Experiment 3 complete.\n")


if __name__ == "__main__":
    run_experiment()
