"""
Experiment Master — Universal Benchmark Suite
==============================================
Benchmarks Improved MOPGA against NSGA-II across the complete:
  • ZDT suite   (ZDT1–ZDT4)          — 2-objective, orthogonal
  • DTLZ suite  (DTLZ1–DTLZ4, DTLZ7) — 3-objective, scalable
  • CEC 2009 UF (UF1–UF10)            — 2/3-objective, rotated

Every problem evaluation is wrapped in try/except so one failure never
kills the entire run. Metrics (HV, IGD, Spacing) use a dynamically
computed reference point derived from the true Pareto front.

Usage:
    python experiment_master.py
"""

import numpy as np
import traceback

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from mopga_improved import run_mopga


# =====================================================================
# Configuration
# =====================================================================

POP_SIZE = 100
GENERATIONS = 300
SEED = 1

PROBLEM_NAMES = [
    "uf1", "uf2", "uf3", "uf4", "uf5", "uf6", "uf7",
    "uf8", "uf9", "uf10",
]

# ZDT suite (2-objective)
    #"zdt1", "zdt2", "zdt3", "zdt4",
    # DTLZ suite (3-objective)
    #"dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz7",
    # CEC 2009 UF suite (2- and 3-objective)
# =====================================================================
# CEC 2009 Unconstrained Functions (UF1–UF10)
#
# Reference: Q. Zhang et al., "Multiobjective Optimization Test Instances
#            for the CEC 2009 Special Session and Competition",
#            Technical Report CES-487, University of Essex, 2008.
#
# Convention: All variable indices j below are 1-indexed as in the paper.
#             Python 0-indexed access is written explicitly as (j-1).
# =====================================================================

# ---------- helpers for J-index partitioning ----------

def _j1_j2_2obj(n):
    """Return 1-indexed J1 (odd j) and J2 (even j) for j = 2..n."""
    J1 = list(range(3, n + 1, 2))   # odd j:  3, 5, 7, …
    J2 = list(range(2, n + 1, 2))   # even j: 2, 4, 6, …
    return J1, J2


def _j1_j2_j3_3obj(n):
    """Return 1-indexed J1, J2, J3 for j = 3..n (mod 3 partition)."""
    J1, J2, J3 = [], [], []
    for j in range(3, n + 1):
        r = j % 3
        if r == 1:
            J1.append(j)
        elif r == 2:
            J2.append(j)
        else:
            J3.append(j)
    return J1, J2, J3


# =====================================================================
# UF1
# =====================================================================

class UF1(Problem):
    """CEC 2009 UF1.  2-obj, x1 in [0,1], xj in [-1,1]."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0); xl[0] = 0.0
        xu = np.full(n_var,  1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        J1, J2 = _j1_j2_2obj(n)
        s1 = sum((X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n))**2 for j in J1)
        s2 = sum((X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n))**2 for j in J2)
        f1 = x1 + 2.0/len(J1) * s1
        f2 = 1.0 - np.sqrt(x1) + 2.0/len(J2) * s2
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1 - np.sqrt(f1)])


# =====================================================================
# UF2
# =====================================================================

class UF2(Problem):
    """CEC 2009 UF2.  2-obj, x1 in [0,1], xj in [-1,1]."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0); xl[0] = 0.0
        xu = np.full(n_var,  1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        J1, J2 = _j1_j2_2obj(n)

        def _base(j):
            return (0.3*x1**2*np.cos(24*np.pi*x1 + 4*j*np.pi/n) + 0.6*x1)

        s1 = sum((X[:, j-1] - _base(j)*np.sin(6*np.pi*x1 + j*np.pi/n))**2 for j in J1)
        s2 = sum((X[:, j-1] - _base(j)*np.cos(6*np.pi*x1 + j*np.pi/n))**2 for j in J2)
        f1 = x1 + 2.0/len(J1) * s1
        f2 = 1.0 - np.sqrt(x1) + 2.0/len(J2) * s2
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1 - np.sqrt(f1)])


# =====================================================================
# UF3
# =====================================================================

class UF3(Problem):
    """CEC 2009 UF3.  2-obj, xj in [0,1]."""

    def __init__(self, n_var=30):
        super().__init__(n_var=n_var, n_obj=2,
                         xl=np.zeros(n_var), xu=np.ones(n_var))

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        J1, J2 = _j1_j2_2obj(n)

        def _rastrigin_sum(Jk):
            s = np.zeros(len(X)); prod = np.ones(len(X))
            for j in Jk:
                exp = 0.5 * (1.0 + 3.0*(j - 2.0)/(n - 2.0))
                yj = X[:, j-1] - np.power(np.maximum(x1, 1e-30), exp)
                s += yj**2
                prod *= np.cos(20.0 * yj * np.pi / np.sqrt(j))
            return 4.0*s - 2.0*prod + 2.0

        f1 = x1 + 2.0/len(J1) * _rastrigin_sum(J1)
        f2 = 1.0 - np.sqrt(x1) + 2.0/len(J2) * _rastrigin_sum(J2)
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1 - np.sqrt(f1)])


# =====================================================================
# UF4
# =====================================================================

class UF4(Problem):
    """CEC 2009 UF4.  2-obj, x1 in [0,1], xj in [-2,2]."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -2.0); xl[0] = 0.0
        xu = np.full(n_var,  2.0); xu[0] = 1.0
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        J1, J2 = _j1_j2_2obj(n)

        def _h(t):
            at = np.abs(t)
            return at / (1.0 + np.exp(2.0 * at))

        s1 = sum(_h(X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n)) for j in J1)
        s2 = sum(_h(X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n)) for j in J2)
        f1 = x1 + 2.0/len(J1) * s1
        f2 = 1.0 - x1**2 + 2.0/len(J2) * s2
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1 - f1**2])


# =====================================================================
# UF5
# =====================================================================

class UF5(Problem):
    """CEC 2009 UF5.  2-obj, x1 in [0,1], xj in [-1,1].  Discrete PF."""

    def __init__(self, n_var=30, N=10, eps=0.1):
        xl = np.full(n_var, -1.0); xl[0] = 0.0
        xu = np.full(n_var,  1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)
        self.N = N; self.eps = eps

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        N, eps = self.N, self.eps
        J1, J2 = _j1_j2_2obj(n)

        def _h(t):
            return 2.0*t**2 - np.cos(4.0*np.pi*t) + 1.0

        s1 = sum(_h(X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n)) for j in J1)
        s2 = sum(_h(X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n)) for j in J2)
        penalty = (1.0/(2*N) + eps) * np.abs(np.sin(2*N*np.pi*x1))
        f1 = x1 + penalty + 2.0/len(J1) * s1
        f2 = 1.0 - x1 + penalty + 2.0/len(J2) * s2
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, **kwargs):
        # 2N+1 discrete points on f1 + f2 = 1
        N = self.N
        f1 = np.linspace(0, 1, 2*N + 1)
        return np.column_stack([f1, 1.0 - f1])


# =====================================================================
# UF6
# =====================================================================

class UF6(Problem):
    """CEC 2009 UF6.  2-obj, x1 in [0,1], xj in [-1,1].  Disconnected PF."""

    def __init__(self, n_var=30, N=2, eps=0.1):
        xl = np.full(n_var, -1.0); xl[0] = 0.0
        xu = np.full(n_var,  1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)
        self.N = N; self.eps = eps

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        N, eps = self.N, self.eps
        J1, J2 = _j1_j2_2obj(n)

        def _rastrigin_sum(Jk):
            s = np.zeros(len(X)); prod = np.ones(len(X))
            for j in Jk:
                yj = X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n)
                s += yj**2
                prod *= np.cos(20.0*yj*np.pi / np.sqrt(j))
            return 4.0*s - 2.0*prod + 2.0

        penalty = np.maximum(0.0, 2.0*(1.0/(2*N) + eps)*np.sin(2*N*np.pi*x1))
        f1 = x1 + penalty + 2.0/len(J1) * _rastrigin_sum(J1)
        f2 = 1.0 - x1 + penalty + 2.0/len(J2) * _rastrigin_sum(J2)
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        # PF segments where sin(2*N*pi*x1) <= 0
        N = self.N
        pts = []
        for x1 in np.linspace(0, 1, n_points * 5):
            if np.sin(2*N*np.pi*x1) <= 1e-6:
                pts.append([x1, 1.0 - x1])
        return np.array(pts) if pts else np.column_stack([
            np.linspace(0, 1, n_points), 1 - np.linspace(0, 1, n_points)
        ])


# =====================================================================
# UF7
# =====================================================================

class UF7(Problem):
    """CEC 2009 UF7.  2-obj, x1 in [0,1], xj in [-1,1].  Linear PF."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0); xl[0] = 0.0
        xu = np.full(n_var,  1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]
        J1, J2 = _j1_j2_2obj(n)
        s1 = sum((X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n))**2 for j in J1)
        s2 = sum((X[:, j-1] - np.sin(6*np.pi*x1 + j*np.pi/n))**2 for j in J2)
        f1 = np.power(np.maximum(x1, 1e-30), 0.2) + 2.0/len(J1) * s1
        f2 = 1.0 - np.power(np.maximum(x1, 1e-30), 0.2) + 2.0/len(J2) * s2
        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500):
        f1 = np.linspace(0, 1, n_points)
        return np.column_stack([f1, 1.0 - f1])


# =====================================================================
# UF8  (3-objective)
# =====================================================================

class UF8(Problem):
    """CEC 2009 UF8.  3-obj, x1,x2 in [0,1], xj in [-2,2]."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -2.0); xl[0] = 0.0; xl[1] = 0.0
        xu = np.full(n_var,  2.0); xu[0] = 1.0; xu[1] = 1.0
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]; x2 = X[:, 1]
        J1, J2, J3 = _j1_j2_j3_3obj(n)
        def _yj(j):
            return X[:, j-1] - 2.0*x2*np.sin(2*np.pi*x1 + j*np.pi/n)
        s1 = sum(_yj(j)**2 for j in J1) if J1 else 0.0
        s2 = sum(_yj(j)**2 for j in J2) if J2 else 0.0
        s3 = sum(_yj(j)**2 for j in J3) if J3 else 0.0
        f1 = np.cos(0.5*np.pi*x1)*np.cos(0.5*np.pi*x2) + 2.0/max(len(J1),1)*s1
        f2 = np.cos(0.5*np.pi*x1)*np.sin(0.5*np.pi*x2) + 2.0/max(len(J2),1)*s2
        f3 = np.sin(0.5*np.pi*x1) + 2.0/max(len(J3),1)*s3
        out["F"] = np.column_stack([f1, f2, f3])

    def pareto_front(self, n_points=50):
        # First octant of unit sphere
        pts = []
        for a in np.linspace(0, np.pi/2, n_points):
            for b in np.linspace(0, np.pi/2, n_points):
                pts.append([np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)])
        return np.array(pts)


# =====================================================================
# UF9  (3-objective)
# =====================================================================

class UF9(Problem):
    """CEC 2009 UF9.  3-obj, x1,x2 in [0,1], xj in [-2,2].  Discontinuous PF."""

    def __init__(self, n_var=30, eps=0.1):
        xl = np.full(n_var, -2.0); xl[0] = 0.0; xl[1] = 0.0
        xu = np.full(n_var,  2.0); xu[0] = 1.0; xu[1] = 1.0
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)
        self.eps = eps

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]; x2 = X[:, 1]; eps = self.eps
        J1, J2, J3 = _j1_j2_j3_3obj(n)
        def _yj(j):
            return X[:, j-1] - 2.0*x2*np.sin(2*np.pi*x1 + j*np.pi/n)
        s1 = sum(_yj(j)**2 for j in J1) if J1 else 0.0
        s2 = sum(_yj(j)**2 for j in J2) if J2 else 0.0
        s3 = sum(_yj(j)**2 for j in J3) if J3 else 0.0
        g = np.maximum(0.0, (1.0 + eps)*(1.0 - 4.0*(2.0*x1 - 1.0)**2))
        f1 = 0.5*(g + 2.0*x1)*x2 + 2.0/max(len(J1),1)*s1
        f2 = 0.5*(g + 2.0 - 2.0*x1)*x2 + 2.0/max(len(J2),1)*s2
        f3 = 1.0 - x2 + 2.0/max(len(J3),1)*s3
        out["F"] = np.column_stack([f1, f2, f3])

    def pareto_front(self, n_points=80):
        eps = self.eps; pts = []
        for x1 in np.linspace(0, 1, n_points):
            g = max(0.0, (1.0 + eps)*(1.0 - 4.0*(2.0*x1 - 1.0)**2))
            for x2 in np.linspace(0, 1, n_points):
                f1 = 0.5*(g + 2.0*x1)*x2
                f2 = 0.5*(g + 2.0 - 2.0*x1)*x2
                f3 = 1.0 - x2
                pts.append([f1, f2, f3])
        return np.array(pts)


# =====================================================================
# UF10  (3-objective)
# =====================================================================

class UF10(Problem):
    """CEC 2009 UF10.  3-obj, x1,x2 in [0,1], xj in [-2,2]."""

    def __init__(self, n_var=30):
        xl = np.full(n_var, -2.0); xl[0] = 0.0; xl[1] = 0.0
        xu = np.full(n_var,  2.0); xu[0] = 1.0; xu[1] = 1.0
        super().__init__(n_var=n_var, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var; x1 = X[:, 0]; x2 = X[:, 1]
        J1, J2, J3 = _j1_j2_j3_3obj(n)
        def _yj(j):
            return X[:, j-1] - 2.0*x2*np.sin(2*np.pi*x1 + j*np.pi/n)
        def _h(t):
            return 4.0*t**2 - np.cos(8.0*np.pi*t) + 1.0
        s1 = sum(_h(_yj(j)) for j in J1) if J1 else 0.0
        s2 = sum(_h(_yj(j)) for j in J2) if J2 else 0.0
        s3 = sum(_h(_yj(j)) for j in J3) if J3 else 0.0
        f1 = np.cos(0.5*np.pi*x1)*np.cos(0.5*np.pi*x2) + 2.0/max(len(J1),1)*s1
        f2 = np.cos(0.5*np.pi*x1)*np.sin(0.5*np.pi*x2) + 2.0/max(len(J2),1)*s2
        f3 = np.sin(0.5*np.pi*x1) + 2.0/max(len(J3),1)*s3
        out["F"] = np.column_stack([f1, f2, f3])

    def pareto_front(self, n_points=50):
        # Same as UF8: first octant of unit sphere
        pts = []
        for a in np.linspace(0, np.pi/2, n_points):
            for b in np.linspace(0, np.pi/2, n_points):
                pts.append([np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)])
        return np.array(pts)


# =====================================================================
# UF Problem Registry
# =====================================================================

UF_REGISTRY = {
    "uf1": UF1, "uf2": UF2, "uf3": UF3, "uf4": UF4, "uf5": UF5,
    "uf6": UF6, "uf7": UF7, "uf8": UF8, "uf9": UF9, "uf10": UF10,
}


# =====================================================================
# Problem Loader
# =====================================================================

def load_problem(name):
    """
    Load a benchmark problem by name and return (problem, pareto_front).
    Handles ZDT, DTLZ (with n_obj=3), and custom CEC 2009 UF problems.
    """
    name_lower = name.lower()

    # --- CEC 2009 UF suite (custom implementations) ---
    if name_lower in UF_REGISTRY:
        problem = UF_REGISTRY[name_lower]()
        pf = problem.pareto_front()
        return problem, pf

    # --- DTLZ suite: 3-objective ---
    if "dtlz" in name_lower:
        problem = get_problem(name_lower, n_obj=3)
    else:
        problem = get_problem(name_lower)

    # Retrieve the true Pareto front
    pf = None
    try:
        pf = problem.pareto_front()
    except TypeError:
        # Some pymoo versions need ref_dirs for DTLZ PF
        try:
            from pymoo.util.ref_dirs import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", problem.n_obj,
                                                n_partitions=12)
            pf = problem.pareto_front(ref_dirs)
        except Exception:
            pass
    except Exception:
        pass

    if pf is None or len(pf) == 0:
        print(f"  [WARN] Could not load Pareto front for {name}.")

    return problem, pf


# =====================================================================
# Metrics
# =====================================================================

def compute_spacing(F):
    """
    Compute the Spacing indicator for a set of non-dominated solutions.

    Spacing measures how uniformly the solutions are distributed along the
    obtained Pareto front.  A lower value indicates more uniform spread.

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
    return float(np.sqrt(np.mean((d - d_mean) ** 2)))


def compute_metrics(F, pf):
    """
    Compute HV, IGD, and Spacing for a front F against the true Pareto front pf.
    The HV reference point is computed dynamically as max(pf) * 1.1.
    Returns a dict with 'HV', 'IGD', and 'SP' keys.
    """
    result = {"HV": float("nan"), "IGD": float("nan"), "SP": float("nan")}

    if F is None or len(F) == 0 or pf is None or len(pf) == 0:
        return result

    try:
        ref_point = np.max(pf, axis=0) * 1.1
        hv = HV(ref_point=ref_point)
        result["HV"] = float(hv(F))
    except Exception as e:
        print(f"    [HV error] {e}")

    try:
        igd = IGD(pf)
        result["IGD"] = float(igd(F))
    except Exception as e:
        print(f"    [IGD error] {e}")

    try:
        result["SP"] = compute_spacing(F)
    except Exception as e:
        print(f"    [SP error] {e}")

    return result


# =====================================================================
# Runners
# =====================================================================

def run_nsga2_benchmark(problem):
    """Run pymoo NSGA-II and return objective values F."""
    algorithm = NSGA2(pop_size=POP_SIZE)
    res = minimize(
        problem, algorithm,
        termination=("n_gen", GENERATIONS),
        seed=SEED, verbose=False,
    )
    return res.F


def run_mopga_benchmark(problem):
    """Run our Improved MOPGA and return objective values F."""
    return run_mopga(problem, pop_size=POP_SIZE, generations=GENERATIONS, seed=SEED)


# =====================================================================
# Output Formatting
# =====================================================================

def print_table(name, n2_m, mo_m):
    """Print a cleanly formatted academic-style comparison table."""
    w = 66
    n_obj_label = "3-obj" if "dtlz" in name or name in ("uf8","uf9","uf10") else "2-obj"
    print()
    print("+" + "-"*(w-2) + "+")
    print(f"|  {name.upper():10s}  ({n_obj_label}){'':>35s}   |")
    print("+" + "-"*(w-2) + "+")
    print(f"|  {'Metric':<10s} | {'NSGA-II':>18s} | {'MOPGA Improved':>18s} |")
    print("|" + "-"*(w-2) + "|")

    for key in ("HV", "IGD", "SP"):
        nv = n2_m[key]; mv = mo_m[key]

        nv_s = f"{nv:.6f}" if not np.isnan(nv) else "      N/A"
        mv_s = f"{mv:.6f}" if not np.isnan(mv) else "      N/A"

        # Determine winner (HV: higher is better; IGD, SP: lower is better)
        if not (np.isnan(nv) or np.isnan(mv)):
            if key == "HV":
                w_n = " *" if nv > mv else "  "
                w_m = " *" if mv > nv else "  "
            else:
                w_n = " *" if nv < mv else "  "
                w_m = " *" if mv < nv else "  "
        else:
            w_n = w_m = "  "

        print(f"|  {key:<10s} | {nv_s:>16s}{w_n} | {mv_s:>16s}{w_m} |")

    print("+" + "-"*(w-2) + "+")
    print("|  * = better for that metric" + " "*(w - 30) + "|")
    print("+" + "-"*(w-2) + "+")


def print_final_summary(all_results):
    """Print a grand summary table of all problems."""
    w = 100
    print("\n\n" + "=" * w)
    print("  GRAND SUMMARY — MOPGA vs NSGA-II")
    print("=" * w)
    print(f"  {'Problem':<10s} | {'HV(NSGA)':>10s}  {'HV(MOPGA)':>10s}  "
          f"{'IGD(NSGA)':>10s}  {'IGD(MOPGA)':>10s}  "
          f"{'SP(NSGA)':>9s}  {'SP(MOPGA)':>9s} | {'Winner':>8s}")
    print("-" * w)

    mopga_wins = 0; nsga_wins = 0; ties = 0

    for name, (n2, mo) in all_results.items():
        hv_n = f"{n2['HV']:.4f}" if not np.isnan(n2['HV']) else "N/A"
        hv_m = f"{mo['HV']:.4f}" if not np.isnan(mo['HV']) else "N/A"
        ig_n = f"{n2['IGD']:.4f}" if not np.isnan(n2['IGD']) else "N/A"
        ig_m = f"{mo['IGD']:.4f}" if not np.isnan(mo['IGD']) else "N/A"
        sp_n = f"{n2['SP']:.4f}" if not np.isnan(n2['SP']) else "N/A"
        sp_m = f"{mo['SP']:.4f}" if not np.isnan(mo['SP']) else "N/A"

        # Score: +1 for each metric won (HV higher=better, IGD/SP lower=better)
        score = 0
        if not (np.isnan(n2['HV']) or np.isnan(mo['HV'])):
            score += 1 if mo['HV'] > n2['HV'] else (-1 if n2['HV'] > mo['HV'] else 0)
        if not (np.isnan(n2['IGD']) or np.isnan(mo['IGD'])):
            score += 1 if mo['IGD'] < n2['IGD'] else (-1 if n2['IGD'] < mo['IGD'] else 0)
        if not (np.isnan(n2['SP']) or np.isnan(mo['SP'])):
            score += 1 if mo['SP'] < n2['SP'] else (-1 if n2['SP'] < mo['SP'] else 0)

        if score > 0:
            winner = "MOPGA"; mopga_wins += 1
        elif score < 0:
            winner = "NSGA-II"; nsga_wins += 1
        else:
            winner = "TIE"; ties += 1

        print(f"  {name.upper():<10s} | {hv_n:>10s}  {hv_m:>10s}  "
              f"{ig_n:>10s}  {ig_m:>10s}  "
              f"{sp_n:>9s}  {sp_m:>9s} | {winner:>8s}")

    print("-" * w)
    total = mopga_wins + nsga_wins + ties
    print(f"  MOPGA wins: {mopga_wins}/{total}    "
          f"NSGA-II wins: {nsga_wins}/{total}    "
          f"Ties: {ties}/{total}")
    print("=" * w)


# =====================================================================
# Main Experiment Loop
# =====================================================================

def main():
    print("=" * 66)
    print("  EXPERIMENT MASTER — Universal Benchmark Suite")
    print(f"  Pop: {POP_SIZE} | Gen: {GENERATIONS} | Seed: {SEED}")
    print(f"  Suites: ZDT(4) + DTLZ(5) + CEC-2009-UF(10) = {len(PROBLEM_NAMES)} problems")
    print("=" * 66)

    all_results = {}

    for idx, name in enumerate(PROBLEM_NAMES, 1):
        print(f"\n{'='*66}")
        print(f"  [{idx}/{len(PROBLEM_NAMES)}] {name.upper()}")
        print(f"{'='*66}")

        try:
            # Load
            problem, pf = load_problem(name)
            if pf is None or len(pf) == 0:
                print(f"  [SKIP] No Pareto front available for {name}.")
                continue

            # Run NSGA-II
            print(f"  Running NSGA-II ...")
            nsga2_F = run_nsga2_benchmark(problem)
            print(f"  NSGA-II done  ({len(nsga2_F)} solutions, "
                  f"{problem.n_obj}-obj)")

            # Run MOPGA
            print(f"  Running MOPGA Improved ...")
            mopga_F = run_mopga_benchmark(problem)
            print(f"  MOPGA done    ({len(mopga_F)} solutions)")

            # Metrics
            nsga2_metrics = compute_metrics(nsga2_F, pf)
            mopga_metrics = compute_metrics(mopga_F, pf)

            # Print per-problem table
            print_table(name, nsga2_metrics, mopga_metrics)

            all_results[name] = (nsga2_metrics, mopga_metrics)

        except Exception:
            print(f"\n  [ERROR] {name.upper()} failed:")
            traceback.print_exc()
            print("  Continuing to next problem ...\n")

    # Grand summary
    if all_results:
        print_final_summary(all_results)

    print("\n[OK] Experiment Master complete.\n")


if __name__ == "__main__":
    main()
