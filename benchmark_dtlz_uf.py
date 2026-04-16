"""
================================================================================
Comprehensive Benchmarking Suite:
  NSGA-II  vs.  Anchored Alternating-Phase MOPGA (Improved)

Test Suites:
  - DTLZ1, DTLZ2  -- 3 objectives
  - UF1, UF2       -- 2 objectives (CEC 2009 unconstrained)

Metrics:
  Hypervolume (HV), Inverted Generational Distance (IGD), Spacing (SP)

Visualization:
  3-D scatter for DTLZ, 2-D scatter for UF

Author  : Postdoctoral Researcher -- Evolutionary Computation & Swarm Intelligence
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (enables 3-D projection)

# -- pymoo imports --
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.core.problem import Problem


# =========================================================================
#  SECTION 0 -- CEC 2009 UF PROBLEM DEFINITIONS
#
#  UF1 and UF2 are NOT included in pymoo 0.6.x, so we implement them
#  directly from the CEC 2009 competition technical report:
#    Q. Zhang et al., "Multiobjective optimization Test Instances for
#    the CEC 2009 Special Session and Competition," 2008.
# =========================================================================

class UF1(Problem):
    """
    CEC 2009 Unconstrained Problem UF1 (2 objectives, n_var = 30).

    x1 in [0, 1], x_j in [-1, 1] for j = 2, ..., n.
    True Pareto front: f2 = 1 - sqrt(f1), with f1 in [0, 1].
    """

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0)
        xl[0] = 0.0
        xu = np.full(n_var, 1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var
        N = X.shape[0]
        x1 = X[:, 0]

        # Odd-indexed variables (j = 2, 4, 6, ... i.e. Python index 1, 3, 5, ...)
        # Even-indexed variables (j = 3, 5, 7, ... i.e. Python index 2, 4, 6, ...)
        # Note: CEC 2009 uses 1-based indexing; j odd means j=3,5,...  j even means j=2,4,...
        J1 = []  # j is odd  (1-based)
        J2 = []  # j is even (1-based)
        for j in range(2, n + 1):  # j from 2 to n (1-based)
            if j % 2 == 1:
                J1.append(j - 1)  # convert to 0-based
            else:
                J2.append(j - 1)

        # Compute sum1 (odd j) and sum2 (even j)
        sum1 = np.zeros(N)
        for j_idx in J1:
            j = j_idx + 1  # back to 1-based for formula
            val = X[:, j_idx] - np.sin(6.0 * np.pi * x1 + j * np.pi / n)
            sum1 += val ** 2

        sum2 = np.zeros(N)
        for j_idx in J2:
            j = j_idx + 1
            val = X[:, j_idx] - np.sin(6.0 * np.pi * x1 + j * np.pi / n)
            sum2 += val ** 2

        f1 = x1 + 2.0 * sum1 / max(len(J1), 1)
        f2 = 1.0 - np.sqrt(x1) + 2.0 * sum2 / max(len(J2), 1)

        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500, **kwargs):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class UF2(Problem):
    """
    CEC 2009 Unconstrained Problem UF2 (2 objectives, n_var = 30).

    x1 in [0, 1], x_j in [-1, 1] for j = 2, ..., n.
    True Pareto front: f2 = 1 - sqrt(f1), with f1 in [0, 1].
    """

    def __init__(self, n_var=30):
        xl = np.full(n_var, -1.0)
        xl[0] = 0.0
        xu = np.full(n_var, 1.0)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        n = self.n_var
        N = X.shape[0]
        x1 = X[:, 0]

        J1 = []  # j odd  (1-based)
        J2 = []  # j even (1-based)
        for j in range(2, n + 1):
            if j % 2 == 1:
                J1.append(j - 1)
            else:
                J2.append(j - 1)

        # UF2 uses different y_j depending on parity:
        #   For j even: y_j = x_j - (0.3*x1^2 * cos(24*pi*x1 + 4*j*pi/n) + 0.6*x1) * cos(6*pi*x1 + j*pi/n)
        #   For j odd:  y_j = x_j - (0.3*x1^2 * cos(24*pi*x1 + 4*j*pi/n) + 0.6*x1) * sin(6*pi*x1 + j*pi/n)

        sum1 = np.zeros(N)
        for j_idx in J1:
            j = j_idx + 1
            y_j = X[:, j_idx] - (
                0.3 * x1 ** 2 * np.cos(24.0 * np.pi * x1 + 4.0 * j * np.pi / n) + 0.6 * x1
            ) * np.sin(6.0 * np.pi * x1 + j * np.pi / n)
            sum1 += y_j ** 2

        sum2 = np.zeros(N)
        for j_idx in J2:
            j = j_idx + 1
            y_j = X[:, j_idx] - (
                0.3 * x1 ** 2 * np.cos(24.0 * np.pi * x1 + 4.0 * j * np.pi / n) + 0.6 * x1
            ) * np.cos(6.0 * np.pi * x1 + j * np.pi / n)
            sum2 += y_j ** 2

        f1 = x1 + 2.0 * sum1 / max(len(J1), 1)
        f2 = 1.0 - np.sqrt(x1) + 2.0 * sum2 / max(len(J2), 1)

        out["F"] = np.column_stack([f1, f2])

    def pareto_front(self, n_points=500, **kwargs):
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


# =========================================================================
#  SECTION 1 -- UTILITY FUNCTIONS (sorting, crowding, evaluation)
# =========================================================================

def dominates(a, b):
    """Return True if objective vector *a* Pareto-dominates *b*."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    """
    Perform fast non-dominated sorting.

    Returns
    -------
    fronts : list[list[int]]
        Each sub-list contains the indices of one Pareto front (0 = best).
    ranks  : np.ndarray
        Front rank for every individual.
    """
    n = F.shape[0]
    S = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    ranks = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts, ranks


def crowding_distance(F, front):
    """Compute crowding distance for a single non-dominated front."""
    size = len(front)
    distances = np.zeros(size, dtype=float)
    if size == 0:
        return distances

    front_F = F[np.array(front, dtype=int)]
    n_obj = front_F.shape[1]

    for m in range(n_obj):
        order = np.argsort(front_F[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        denominator = front_F[order[-1], m] - front_F[order[0], m]
        if denominator <= 0:
            continue
        for j in range(1, size - 1):
            distances[order[j]] += (
                front_F[order[j + 1], m] - front_F[order[j - 1], m]
            ) / denominator

    return distances


def evaluate_population(problem, X):
    """Evaluate a population through pymoo's problem interface."""
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.population import Population

    pop = Population.new("X", X)
    Evaluator().eval(problem, pop)
    return pop.get("F")


def initialize_population(pop_size, problem):
    """Uniform random initialisation within variable bounds."""
    lower = np.asarray(problem.xl, dtype=float)
    upper = np.asarray(problem.xu, dtype=float)
    n_var = problem.n_var

    X = np.random.rand(pop_size, n_var)
    X = lower + X * (upper - lower)
    F = evaluate_population(problem, X)
    return X, F


# =========================================================================
#  SECTION 2 -- MOPGA COMPONENT OPERATORS
# =========================================================================

# -- 2a. Light / Shade partitioning ------------------------------------------

def divide_light_shade(ranks, crowding):
    """Split population into light (exploitation) and shade (exploration)."""
    score = ranks + 1.0 / (crowding + 1.0)
    sorted_indices = np.argsort(score)
    cutoff = len(score) // 2
    light_mask = np.zeros(len(score), dtype=bool)
    light_mask[sorted_indices[:cutoff]] = True
    return light_mask, ~light_mask


# -- 2b. Gaussian mutation for Growth Phase -----------------------------------

def mutate_solution(x, lower, upper, strength):
    """Gaussian perturbation clipped to bounds."""
    noise = np.random.normal(loc=0.0, scale=strength, size=x.shape)
    child = x + noise * (upper - lower)
    return np.clip(child, lower, upper)


# -- 2c. Island-Aware Magnetic Repulsion --------------------------------------

def cell_vicinity_repulsion(x, population, radius, lower, upper):
    """
    Distance-gated repulsion (island-aware magnetic push).

    Algorithm
    ---------
    1. Sample up to 15 random peers.
    2. Compute distances; ignore zeros (self-measurement).
    3. Threshold = median(distances) * 0.5.
    4. Collect close_peers within threshold.
       - If none -> standing on an island cliff -> do NOT push.
       - If some -> push EXACTLY AWAY from the mean of close_peers, scaled by radius.
    """
    if len(population) < 2:
        return x

    # Step 1: sample local peers
    sample_size = min(15, len(population))
    idx = np.random.choice(len(population), sample_size, replace=False)
    peers = population[idx]

    # Step 2: physical distances (ignore self / zero)
    distances = np.linalg.norm(peers - x, axis=1)
    nonzero_mask = distances > 1e-9
    if not np.any(nonzero_mask):
        return x                                   # all peers coincide -> no push

    valid_distances = distances[nonzero_mask]

    # Step 3: threshold
    threshold = np.median(valid_distances) * 0.5

    # Step 4: close peers
    close_mask = (distances < threshold) & nonzero_mask
    close_peers = peers[close_mask]

    if len(close_peers) == 0:
        return x                                   # island cliff -- no push

    # Normalised repulsion vector
    direction = x - np.mean(close_peers, axis=0)
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        direction = direction / norm

    candidate = x + radius * direction
    return np.clip(candidate, lower, upper)


# -- 2d. Tournament selection -------------------------------------------------

def tournament_selection(indices, crowding_vals, pool_size=2):
    """Binary tournament on crowding distance within front-0."""
    if len(indices) == 0:
        return 0
    candidates = np.random.choice(indices, size=pool_size, replace=True)
    best = candidates[0]
    for c in candidates[1:]:
        if crowding_vals[c] > crowding_vals[best]:
            best = c
    return best


# -- 2e. SBX crossover --------------------------------------------------------

def sbx_crossover(p1, p2, lower, upper, prob=0.9, eta_c=15):
    """Simulated Binary Crossover between two parents."""
    child = p1.copy()
    if np.random.rand() > prob:
        return child
    for j in range(len(p1)):
        if np.random.rand() > 0.5:
            continue
        if abs(p1[j] - p2[j]) < 1e-14:
            continue
        y1, y2 = min(p1[j], p2[j]), max(p1[j], p2[j])
        yl, yu = lower[j], upper[j]
        rand = np.random.rand()
        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1 + 1e-14))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        child[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        child[j] = np.clip(child[j], yl, yu)
    return child


# -- 2f. Polynomial mutation ---------------------------------------------------

def polynomial_mutation(x, lower, upper, prob, eta_m=20):
    """Polynomial mutation for a single individual."""
    child = x.copy()
    for j in range(len(x)):
        if np.random.rand() > prob:
            continue
        y = child[j]
        yl, yu = lower[j], upper[j]
        delta = yu - yl
        if delta < 1e-14:
            continue
        r = np.random.rand()
        if r < 0.5:
            deltaq = (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            deltaq = 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0))
        child[j] = y + deltaq * delta
        child[j] = np.clip(child[j], yl, yu)
    return child


# -- 2g. Selection by Pareto rank + crowding -----------------------------------

def select_next_population(X, F, pop_size):
    """Environmental selection: fill by front rank, then crowding distance."""
    fronts, _ = fast_non_dominated_sort(F)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
            continue
        distances = crowding_distance(F, front)
        order = np.argsort(-distances)
        slots = pop_size - len(selected)
        selected.extend([front[i] for i in order[:slots]])
        break

    idx = np.array(selected, dtype=int)
    return X[idx], F[idx]


# =========================================================================
#  SECTION 3 -- ANCHORED ALTERNATING-PHASE MOPGA (MAIN ENGINE)
# =========================================================================

def run_mopga(problem, pop_size, generations, seed=42):
    """
    Anchored Alternating-Phase Multi-Objective Phototropic Growth Algorithm.

    Key mechanics
    -------------
    1. **Alternating phases** (gen % 5 controls the switch):
       - Growth Phase  -> continuous phototropic vectors (auxin, light/shade).
       - Pollination Phase -> tournament + SBX + polynomial mutation.

    2. **M-dimensional anchor** (CRITICAL):
       For an M-objective problem the first (M-1) variables control Pareto
       position/spacing; the remaining variables control depth/distance.
       During Pollination, the child inherits the parent's first (M-1)
       variables EXACTLY.  Only the distance variables are mutated.

    3. **Island-aware magnetic repulsion** in the Growth Phase replaces
       plain vicinity cohesion with a distance-gated push-away from
       the local centre of mass.

    Parameters
    ----------
    problem    : pymoo Problem instance
    pop_size   : int
    generations: int
    seed       : int

    Returns
    -------
    (X, F) -- decision variables and objective values of the final
              non-dominated set.
    """
    np.random.seed(seed)
    lower = np.asarray(problem.xl, dtype=float)
    upper = np.asarray(problem.xu, dtype=float)
    n_var = problem.n_var
    n_obj = problem.n_obj                          # M (number of objectives)

    # The first (M-1) variables are the "anchor" (position/spacing)
    anchor_dim = n_obj - 1

    X, F = initialize_population(pop_size, problem)

    for gen in range(generations):
        # -- Non-dominated sorting + crowding for current population --
        fronts, ranks = fast_non_dominated_sort(F)
        crowding_vals = np.zeros(len(F), dtype=float)
        for front in fronts:
            cd = crowding_distance(F, front)
            crowding_vals[np.array(front, dtype=int)] = cd

        offspring = np.zeros_like(X)

        # ================================================================
        #  PHASE DECISION
        #    Growth      : gen % 5 != 0  OR  gen >= (generations - 15)
        #    Pollination : otherwise (every 5th generation, not near end)
        # ================================================================

        is_growth_phase = (gen % 5 != 0) or (gen >= generations - 15)

        if is_growth_phase:
            # ============================================================
            #  GROWTH PHASE  -- pure continuous phototropic vectors
            # ============================================================
            alpha = np.exp(-gen / max(1, generations))
            beta = 0.4
            auxin = 0.1 + 0.8 * (1.0 / (1.0 + ranks))

            light_mask, shade_mask = divide_light_shade(ranks, crowding_vals)

            # Leaders from front-0
            front0_idx = np.array(fronts[0], dtype=int) if len(fronts[0]) > 0 else np.arange(len(X))
            leaders = X[front0_idx]

            # Best individual in each group (by rank, then crowding)
            def best_in_group(mask):
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    return X[np.argmin(ranks)]
                best = indices[np.lexsort((-crowding_vals[indices], ranks[indices]))][0]
                return X[best]

            light_best = best_in_group(light_mask)
            shade_best = best_in_group(shade_mask)

            for i in range(len(X)):
                Xi = X[i]
                leader = leaders[np.random.randint(len(leaders))]

                # Base phototropic force-of-competition toward the leader
                foc = alpha * beta * np.random.rand(n_var) * (leader - Xi)

                # Gaussian mutation for local exploration
                strength = 0.05 if light_mask[i] else 0.12
                mutated = mutate_solution(Xi, lower, upper, strength)

                # Directed growth weighted by auxin
                directed = auxin[i] * (leader - mutated)

                if light_mask[i]:
                    local_pop = X[light_mask]
                    light_move = alpha * beta * np.random.rand(n_var) * (light_best - mutated)
                    update = directed + light_move
                else:
                    local_pop = X[shade_mask] if np.any(shade_mask) else X
                    random_peer = X[np.random.randint(len(X))]
                    shade_move = alpha * beta * np.random.rand(n_var) * (random_peer - mutated)
                    shade_best_move = 0.2 * alpha * (shade_best - mutated)
                    update = directed + shade_move + shade_best_move

                # Island-aware magnetic repulsion (replaces plain vicinity)
                vicinity_candidate = cell_vicinity_repulsion(
                    mutated + update, local_pop, alpha * 0.1, lower, upper
                )
                vicinity_term = vicinity_candidate - mutated

                # Final phototropic growth step
                child = Xi + foc + update + vicinity_term
                offspring[i] = np.clip(child, lower, upper)

        else:
            # ============================================================
            #  POLLINATION PHASE  -- genetic exploration
            # ============================================================
            mut_prob = 1.0 / n_var
            front0_indices = np.array(fronts[0], dtype=int)

            for i in range(len(X)):
                Xi = X[i]

                # Tournament selection from front-0
                p1_idx = tournament_selection(front0_indices, crowding_vals)
                p2_idx = tournament_selection(front0_indices, crowding_vals)

                # SBX Crossover + Polynomial Mutation
                child = sbx_crossover(X[p1_idx], X[p2_idx], lower, upper,
                                      prob=0.9, eta_c=15.0)
                child = polynomial_mutation(child, lower, upper,
                                            prob=mut_prob, eta_m=20.0)

                # -- M-DIMENSIONAL ANCHOR (CRITICAL UPGRADE) --
                # Inherit the exact first (M-1) variables from parent Xi.
                # Only the distance variables (indices M-1 ... n_var-1) are
                # kept from the crossover/mutation result.
                child[:anchor_dim] = Xi[:anchor_dim]

                # Clip to bounds AFTER anchoring
                offspring[i] = np.clip(child, lower, upper)

        # -- Merge parent + offspring, then environmental selection --
        F_offspring = evaluate_population(problem, offspring)
        X = np.vstack([X, offspring])
        F = np.vstack([F, F_offspring])
        X, F = select_next_population(X, F, pop_size)

        if gen % 50 == 0:
            print(f"  MOPGA  gen {gen + 1:>4d}/{generations}")

    # Extract final non-dominated set
    pareto_front_indices = fast_non_dominated_sort(F)[0][0]
    return X[np.array(pareto_front_indices, dtype=int)], \
           F[np.array(pareto_front_indices, dtype=int)]


# =========================================================================
#  SECTION 4 -- NSGA-II BASELINE (pymoo)
# =========================================================================

def run_nsga2(problem, pop_size=100, n_gen=250, seed=42):
    """Run pymoo's canonical NSGA-II and return (X, F) of the result set."""
    algorithm = NSGA2(pop_size=pop_size)
    res = pymoo_minimize(
        problem, algorithm, ("n_gen", n_gen),
        seed=seed, save_history=False, verbose=False,
    )
    return res.X, res.F


# =========================================================================
#  SECTION 5 -- METRIC COMPUTATION
# =========================================================================

def compute_hypervolume(F, ref_point):
    """Hypervolume indicator (higher is better)."""
    if F is None or len(F) == 0:
        return 0.0
    F = np.asarray(F, dtype=float)
    try:
        return float(HV(ref_point=ref_point)(F))
    except Exception:
        return 0.0


def compute_igd(F, true_pf):
    """Inverted Generational Distance (lower is better)."""
    if F is None or len(F) == 0:
        return float("inf")
    F = np.asarray(F, dtype=float)
    try:
        return float(IGD(true_pf)(F))
    except Exception:
        return float("inf")


def compute_spacing(F):
    """
    Spacing metric (Schott, 1995).
    Measures uniformity via nearest-neighbour Manhattan distances.
    Lower -> more uniform.
    """
    if F is None or len(F) < 2:
        return 0.0
    F = np.asarray(F, dtype=float)
    n = len(F)
    d = np.zeros(n)
    for i in range(n):
        dists = np.sum(np.abs(F - F[i]), axis=1)
        dists[i] = np.inf
        d[i] = np.min(dists)
    d_mean = np.mean(d)
    return float(np.sqrt(np.sum((d - d_mean) ** 2) / (n - 1)))


# =========================================================================
#  SECTION 6 -- BENCHMARK CONFIGURATION
# =========================================================================

def build_benchmarks():
    """
    Construct the benchmark suite.

    DTLZ problems are obtained from pymoo's registry.
    UF problems use our custom implementations above.
    """
    return [
        # -- DTLZ (3 objectives) --
        {
            "name":      "DTLZ1",
            "problem":   get_problem("dtlz1", n_var=7),   # k=5 -> (M-1)+k = 7
            "ref_point": np.array([400.0, 400.0, 400.0]),  # DTLZ1 PF spans [0, 0.5]
            "plot_dim":  3,
        },
        {
            "name":      "DTLZ2",
            "problem":   get_problem("dtlz2", n_var=12),  # k=10 -> (M-1)+k = 12
            "ref_point": np.array([2.0, 2.0, 2.0]),
            "plot_dim":  3,
        },
        # -- CEC 2009 Unconstrained (2 objectives) --
        {
            "name":      "UF1",
            "problem":   UF1(n_var=30),
            "ref_point": np.array([2.0, 2.0]),
            "plot_dim":  2,
        },
        {
            "name":      "UF2",
            "problem":   UF2(n_var=30),
            "ref_point": np.array([2.0, 2.0]),
            "plot_dim":  2,
        },
    ]


POP_SIZE    = 100
GENERATIONS = 250
SEED        = 42


# =========================================================================
#  SECTION 7 -- VISUALISATION
# =========================================================================

def plot_3d(true_pf, nsga2_F, mopga_F, title, save_path):
    """3-D scatter plot for 3-objective problems (DTLZ)."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # True Pareto Front (grey wireframe look)
    if true_pf is not None and len(true_pf) > 0:
        ax.scatter(
            true_pf[:, 0], true_pf[:, 1], true_pf[:, 2],
            c="grey", s=4, alpha=0.25, label="True PF", depthshade=True,
        )

    # NSGA-II (red)
    if nsga2_F is not None and len(nsga2_F) > 0:
        ax.scatter(
            nsga2_F[:, 0], nsga2_F[:, 1], nsga2_F[:, 2],
            c="red", marker="o", s=28, alpha=0.85, label="NSGA-II",
            edgecolors="darkred", linewidths=0.4,
        )

    # MOPGA (blue)
    if mopga_F is not None and len(mopga_F) > 0:
        ax.scatter(
            mopga_F[:, 0], mopga_F[:, 1], mopga_F[:, 2],
            c="dodgerblue", marker="^", s=32, alpha=0.85, label="MOPGA",
            edgecolors="navy", linewidths=0.4,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("f1", fontsize=11)
    ax.set_ylabel("f2", fontsize=11)
    ax.set_zlabel("f3", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved 3-D plot: {save_path}")
    plt.close()


def plot_2d(true_pf, nsga2_F, mopga_F, title, save_path):
    """2-D scatter plot for 2-objective problems (UF)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # True Pareto Front (grey)
    if true_pf is not None and len(true_pf) > 0:
        ax.scatter(
            true_pf[:, 0], true_pf[:, 1],
            c="grey", s=6, alpha=0.35, label="True PF", zorder=1,
        )

    # NSGA-II (red)
    if nsga2_F is not None and len(nsga2_F) > 0:
        ax.scatter(
            nsga2_F[:, 0], nsga2_F[:, 1],
            c="red", marker="o", s=30, alpha=0.8, label="NSGA-II",
            edgecolors="darkred", linewidths=0.5, zorder=2,
        )

    # MOPGA (blue)
    if mopga_F is not None and len(mopga_F) > 0:
        ax.scatter(
            mopga_F[:, 0], mopga_F[:, 1],
            c="dodgerblue", marker="^", s=34, alpha=0.8, label="MOPGA",
            edgecolors="navy", linewidths=0.5, zorder=3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("f1", fontsize=12)
    ax.set_ylabel("f2", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved 2-D plot: {save_path}")
    plt.close()


# =========================================================================
#  SECTION 8 -- MAIN EXPERIMENT DRIVER
# =========================================================================

def run_benchmark():
    """Execute the full benchmarking experiment."""
    print("=" * 72)
    print("  NSGA-II  vs.  Anchored Alternating-Phase MOPGA")
    print(f"  Population = {POP_SIZE}   |   Generations = {GENERATIONS}   |   Seed = {SEED}")
    print("=" * 72)

    benchmarks = build_benchmarks()
    results = {}

    for bench in benchmarks:
        name      = bench["name"]
        problem   = bench["problem"]
        ref_point = bench["ref_point"]
        plot_dim  = bench["plot_dim"]
        n_obj     = problem.n_obj

        print(f"\n{'-' * 60}")
        print(f"  Benchmark: {name}   ({n_obj} objectives, {problem.n_var} vars)")
        print(f"{'-' * 60}")

        # Obtain true Pareto front (if available)
        try:
            true_pf = problem.pareto_front()
        except Exception:
            true_pf = None

        # -- Run NSGA-II --
        print("  Running NSGA-II ...")
        nsga2_X, nsga2_F = run_nsga2(problem, pop_size=POP_SIZE,
                                      n_gen=GENERATIONS, seed=SEED)

        # -- Run MOPGA --
        print("  Running MOPGA ...")
        mopga_X, mopga_F = run_mopga(problem, pop_size=POP_SIZE,
                                      generations=GENERATIONS, seed=SEED)

        # -- Compute metrics --
        nsga2_hv  = compute_hypervolume(nsga2_F, ref_point)
        nsga2_igd = compute_igd(nsga2_F, true_pf) if true_pf is not None else float("nan")
        nsga2_sp  = compute_spacing(nsga2_F)

        mopga_hv  = compute_hypervolume(mopga_F, ref_point)
        mopga_igd = compute_igd(mopga_F, true_pf) if true_pf is not None else float("nan")
        mopga_sp  = compute_spacing(mopga_F)

        results[name] = {
            "NSGA-II": {"HV": nsga2_hv, "IGD": nsga2_igd, "SP": nsga2_sp},
            "MOPGA":   {"HV": mopga_hv, "IGD": mopga_igd, "SP": mopga_sp},
        }

        # -- Print metric table for this benchmark --
        print(f"\n  {'Metric':<12} {'NSGA-II':>14} {'MOPGA':>14}")
        print(f"  {'-' * 40}")
        print(f"  {'HV':<12} {nsga2_hv:>14.6f} {mopga_hv:>14.6f}")
        print(f"  {'IGD':<12} {nsga2_igd:>14.6f} {mopga_igd:>14.6f}")
        print(f"  {'SP':<12} {nsga2_sp:>14.6f} {mopga_sp:>14.6f}")

        # -- Plot --
        plot_title = f"{name}: NSGA-II vs MOPGA (pop={POP_SIZE}, gen={GENERATIONS})"
        save_path  = f"{name.lower()}_benchmark.png"

        if plot_dim == 3:
            plot_3d(true_pf, nsga2_F, mopga_F, plot_title, save_path)
        else:
            plot_2d(true_pf, nsga2_F, mopga_F, plot_title, save_path)

    # -- Final summary table --
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"\n  {'Problem':<10} {'Algorithm':<12} {'HV':>12} {'IGD':>12} {'SP':>12}")
    print(f"  {'-' * 58}")
    for bname, algos in results.items():
        for aname, metrics in algos.items():
            print(
                f"  {bname:<10} {aname:<12} "
                f"{metrics['HV']:>12.6f} "
                f"{metrics['IGD']:>12.6f} "
                f"{metrics['SP']:>12.6f}"
            )
        print(f"  {'-' * 58}")

    print("\n[OK] Benchmarking complete.\n")
    return results


# =========================================================================
#  ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    run_benchmark()
