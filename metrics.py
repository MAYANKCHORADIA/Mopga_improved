"""
Metric computation module for multi-objective optimization benchmarking.
Provides Hypervolume, Inverted Generational Distance, and Spacing.
"""
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def compute_hypervolume(pareto_front, ref_point=None):
    """
    Compute the Hypervolume indicator for a given Pareto front.

    Parameters
    ----------
    pareto_front : np.ndarray
        Objective values of the approximated Pareto front, shape (n_points, n_obj).
    ref_point : np.ndarray or None
        Reference point for HV. If None, a default of [1.1, 1.1] is used.

    Returns
    -------
    float
        Hypervolume value.
    """
    if pareto_front is None or len(pareto_front) == 0:
        return 0.0
    pareto_front = np.asarray(pareto_front, dtype=float)
    if ref_point is None:
        ref_point = np.array([1.1, 1.1])
    try:
        hv = HV(ref_point=ref_point)
        return float(hv(pareto_front))
    except Exception:
        return 0.0


def compute_igd(pareto_front, problem):
    """
    Compute the Inverted Generational Distance for a given Pareto front.

    Parameters
    ----------
    pareto_front : np.ndarray
        Objective values of the approximated Pareto front.
    problem : pymoo.core.problem.Problem
        The pymoo problem instance (must support .pareto_front()).

    Returns
    -------
    float
        IGD value. Lower is better.
    """
    if pareto_front is None or len(pareto_front) == 0:
        return float("inf")
    pareto_front = np.asarray(pareto_front, dtype=float)
    try:
        true_pf = problem.pareto_front()
        igd = IGD(true_pf)
        return float(igd(pareto_front))
    except Exception:
        return float("inf")


def compute_spacing(pareto_front):
    """
    Compute the Spacing metric for a given Pareto front.

    Measures the uniformity of the distribution of solutions along
    the Pareto front using nearest-neighbor Manhattan distances.

    Parameters
    ----------
    pareto_front : np.ndarray
        Objective values, shape (n_points, n_obj).

    Returns
    -------
    float
        Spacing value. Lower indicates more uniform distribution.
    """
    if pareto_front is None or len(pareto_front) < 2:
        return 0.0
    F = np.asarray(pareto_front, dtype=float)
    n = len(F)

    d = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(F - F[i]), axis=1)
        dists[i] = np.inf
        d[i] = np.min(dists)

    d_mean = np.mean(d)
    spacing = np.sqrt(np.sum((d - d_mean) ** 2) / (n - 1))
    return float(spacing)
