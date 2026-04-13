"""
Plotting utilities for Pareto front visualisation.
"""
import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems import get_problem


def plot_pareto_fronts(algorithm_fronts, problem_name, save_path=None):
    """
    Plot the Pareto fronts of multiple algorithms against the true Pareto front.

    Parameters
    ----------
    algorithm_fronts : dict[str, np.ndarray]
        Mapping of algorithm name -> objective values (n_points, 2).
    problem_name : str
        Name of the ZDT problem (e.g. 'zdt1').
    save_path : str or None
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(10, 8))

    # True Pareto Front
    try:
        problem = get_problem(problem_name, n_var=30)
        true_pf = problem.pareto_front()
        plt.scatter(
            true_pf[:, 0], true_pf[:, 1],
            c="black", s=5, label="True Pareto Front", alpha=0.5, zorder=1,
        )
    except Exception:
        pass

    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    markers = ["o", "x", "+", "s", "D", "^"]

    for idx, (alg_name, F) in enumerate(algorithm_fronts.items()):
        F = np.asarray(F, dtype=float)
        if F.ndim != 2 or F.shape[0] == 0:
            continue
        plt.scatter(
            F[:, 0], F[:, 1],
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=30, label=alg_name, alpha=0.7, zorder=2,
        )

    plt.title(f"{problem_name.upper()} Benchmark: Final Pareto Fronts")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()
