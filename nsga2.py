"""
NSGA-II wrapper using pymoo's built-in implementation.
"""
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def run_nsga2(problem, pop_size=100, n_gen=100, seed=42):
    """
    Run pymoo's NSGA-II and return only the Pareto front objective values.

    Parameters
    ----------
    problem : pymoo.core.problem.Problem
    pop_size : int
    n_gen : int
        Number of generations.
    seed : int

    Returns
    -------
    np.ndarray
        Objective values of the final non-dominated set, shape (n_points, n_obj).
    """
    algorithm = NSGA2(pop_size=pop_size)
    termination = ("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=False,
        verbose=False,
    )

    return res.F
