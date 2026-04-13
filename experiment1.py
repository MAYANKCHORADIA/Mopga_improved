import json
import os
import numpy as np

from problems import get_zdt_problem
from mopga import run_mopga as run_mopga_original
from mopga_improved import run_mopga as run_mopga_improved
from nsga2 import run_nsga2
from metrics import compute_hypervolume, compute_igd, compute_spacing
from plot import plot_pareto_fronts


PROBLEMS = ["zdt1", "zdt2", "zdt3" ,"zdt4"]
#"zdt1", "zdt2", "zdt3" ,"zdt4"
ALGORITHMS = {
    "NSGA-II": run_nsga2,
    "MOPGA Improved": run_mopga_improved,
}
#"MOPGA Original": run_mopga_original,
POP_SIZE = 80
GENERATIONS = 100
RUNS = 5
SEEDS = [101, 202, 303, 404, 505]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_experiment1.json")


def print_summary_table(summary):
    """Print aggregated mean metrics for each problem and algorithm."""
    print("\nBenchmark results for Experiment 1")
    print("=" * 70)

    for problem_name, algorithms in summary.items():
        print(f"\n{problem_name.upper()}")
        for alg_name, data in algorithms.items():
            print(
                f"{alg_name:<18} -> HV: {data['HV']:.6f}, "
                f"IGD: {data['IGD']:.6f}, SP: {data['SP']:.6f}"
            )
    print("=" * 70)


def save_results(summary):
    """Save the experiment summary to a JSON file."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


def run_experiment():
    summary = {}

    for problem_name in PROBLEMS:
        problem = get_zdt_problem(problem_name)
        summary[problem_name] = {}
        print(f"\nRunning experiments on {problem_name.upper()}")

        for algorithm_name, algorithm_fn in ALGORITHMS.items():
            hv_values = []
            igd_values = []
            sp_values = []
            best_pareto = None
            best_hv = -np.inf

            for run_index, seed in enumerate(SEEDS, start=1):
                print(f"{algorithm_name} run {run_index}/{RUNS} (seed={seed})")
                pareto = algorithm_fn(problem, POP_SIZE, GENERATIONS, seed=seed)

                hv = compute_hypervolume(pareto)
                igd = compute_igd(pareto, problem)
                sp = compute_spacing(pareto)

                hv_values.append(hv)
                igd_values.append(igd)
                sp_values.append(sp)

                if hv > best_hv:
                    best_hv = hv
                    best_pareto = pareto

            summary[problem_name][algorithm_name] = {
                "HV": float(np.mean(hv_values)),
                "IGD": float(np.mean(igd_values)),
                "SP": float(np.mean(sp_values)),
                "HV_runs": [float(value) for value in hv_values],
                "IGD_runs": [float(value) for value in igd_values],
                "SP_runs": [float(value) for value in sp_values],
                "pareto": best_pareto.tolist() if best_pareto is not None else [],
            }

    print_summary_table(summary)
    save_results(summary)

    for problem_name in PROBLEMS:
        problem_fronts = {
            alg_name: np.asarray(data.get("pareto", []), dtype=float)
            for alg_name, data in summary[problem_name].items()
        }
        plot_pareto_fronts(problem_fronts, problem_name, save_path=f"{problem_name}_experiment1.png")


if __name__ == "__main__":
    run_experiment()
