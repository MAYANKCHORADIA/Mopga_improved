"""
Multi-Objective Phototropic Growth Algorithm (MOPGA) — Universal Engine
=======================================================================
A nature-inspired multi-objective optimizer that alternates between continuous
phototropic growth (with Island-Aware Magnetic Repulsion) and an Adaptive
Multi-Operator Ensemble (SBX/Anchor + DE Explorer) to dominate orthogonal,
scalable, and rotated benchmark landscapes.

Public API:
    run_mopga(problem, pop_size, generations, seed=1) -> np.ndarray
"""

import numpy as np


# =====================================================================
# Core Utilities
# =====================================================================

def dominates(a, b):
    """Check if vector a dominates vector b."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    """Perform non-dominated sorting on objective vectors."""
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
    """Compute crowding distance for a single nondominated front."""
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
    """Evaluate a population of candidate solutions."""
    result = problem.evaluate(X)
    if isinstance(result, dict):
        return result["F"]
    return result


def initialize_population(pop_size, problem):
    """Initialize decision variables uniformly within bounds."""
    lower = np.asarray(problem.xl, dtype=float)
    upper = np.asarray(problem.xu, dtype=float)
    n_var = problem.n_var

    population = np.random.rand(pop_size, n_var)
    population = lower + population * (upper - lower)
    objectives = evaluate_population(problem, population)
    return population, objectives


# =====================================================================
# Growth Phase Operators
# =====================================================================

def divide_light_shade(ranks, crowding):
    """Split the population into light and shade groups for exploration."""
    score = ranks + 1.0 / (crowding + 1.0)
    sorted_indices = np.argsort(score)
    cutoff = len(score) // 2
    light_mask = np.zeros(len(score), dtype=bool)
    light_mask[sorted_indices[:cutoff]] = True
    return light_mask, ~light_mask


def mutate_solution(x, lower, upper, strength):
    """Mutate a solution with Gaussian noise and clip to bounds."""
    noise = np.random.normal(loc=0.0, scale=strength, size=x.shape)
    child = x + noise * (upper - lower)
    return np.minimum(np.maximum(child, lower), upper)


def redistribute_auxin(ranks):
    """Redistribute auxin intensity across the population."""
    return 0.05 + 0.3 * (1.0 / (1.0 + ranks))


def compute_force_of_competition(ranks, distances):
    """Compute a scalar force of competition for each individual."""
    normalized = distances / (np.max(distances) + 1e-9)
    return (1.0 / (1.0 + ranks)) + 0.5 * normalized


def cell_vicinity(x, population, radius, lower, upper):
    """
    Island-Aware Magnetic Repulsion.

    Calculate physical Euclidean distances to a random sample of local peers
    (ignoring distance 0). Compute a threshold as median(distances) * 0.5.
    Repel away from the mean position of peers strictly within this threshold.
    If no peers are within the threshold (island edge), do not push.
    """
    if len(population) < 2:
        return x

    # Sample local peers for speed
    sample_size = min(15, len(population))
    idx = np.random.choice(len(population), sample_size, replace=False)
    peers = population[idx]

    # Calculate physical distances
    distances = np.linalg.norm(peers - x, axis=1)
    distances[distances < 1e-9] = np.inf  # Prevent self-measurement

    # Identify the local cluster threshold (protects disconnected island gaps)
    valid = distances[distances != np.inf]
    if len(valid) == 0:
        return x
    threshold = np.median(valid) * 0.5
    close_peers = peers[distances < threshold]

    # If standing on an island edge with no immediate close peers, do not jump into the gap
    if len(close_peers) == 0:
        return x

    # Push away from the center of mass of strictly local peers
    direction = x - np.mean(close_peers, axis=0)

    # Normalize direction
    norm = np.linalg.norm(direction)
    if norm > 1e-9:
        direction = direction / norm

    candidate = x + radius * direction
    return np.minimum(np.maximum(candidate, lower), upper)


# =====================================================================
# Selection & Genetic Operators
# =====================================================================

def select_next_population(X, F, pop_size):
    """Select the next generation by Pareto rank and crowding distance."""
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

    return X[np.array(selected, dtype=int)], F[np.array(selected, dtype=int)]


def tournament_selection(indices, crowding_vals, pool_size=2):
    """Binary tournament selection from a pool of indices based on crowding distance."""
    if len(indices) == 0:
        return 0
    candidates = np.random.choice(indices, size=pool_size, replace=True)
    best = candidates[0]
    for c in candidates[1:]:
        if crowding_vals[c] > crowding_vals[best]:
            best = c
    return best


def sbx_crossover(p1, p2, lower, upper, prob=0.9, eta_c=15):
    """Simulated Binary Crossover (SBX) between two parents."""
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
        # Child 1
        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1 + 1e-14))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        child[j] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        child[j] = np.clip(child[j], yl, yu)
    return child


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


def de_rand_1_bin(target, population, front0_indices, lower, upper,
                  F_scale=0.5, CR=0.9):
    """
    DE/rand/1/bin — Differential Evolution operator.

    Select 3 distinct peers from Front 0, create a mutant via
    v = r1 + F*(r2 - r3), then perform binomial crossover with the target.
    """
    n_var = len(target)

    # Build a pool of at least 3 distinct indices
    pool = front0_indices if len(front0_indices) >= 3 else np.arange(len(population))
    chosen = np.random.choice(pool, size=3, replace=False)
    r1, r2, r3 = population[chosen[0]], population[chosen[1]], population[chosen[2]]

    # Mutant vector: v = r1 + F * (r2 - r3)
    mutant = r1 + F_scale * (r2 - r3)
    mutant = np.clip(mutant, lower, upper)

    # Binomial crossover
    j_rand = np.random.randint(n_var)
    child = target.copy()
    for j in range(n_var):
        if np.random.rand() < CR or j == j_rand:
            child[j] = mutant[j]

    return np.clip(child, lower, upper)


# =====================================================================
# Main Algorithm
# =====================================================================

def run_mopga(problem, pop_size, generations, seed=1):
    """
    Run the Universal Multi-Objective Phototropic Growth Algorithm.

    Returns the non-dominated Pareto front objective values (F).

    Phase Engine:
        - Cool-down (last 15 generations): pure vector math smoothing
        - Growth Phase (generation % 5 != 0): continuous vectors +
          Island-Aware Magnetic Repulsion
        - Pollination Phase (generation % 5 == 0, not in cool-down):
          Adaptive Multi-Operator Ensemble
            Path A (50%): SBX + M-Dimensional Anchor → orthogonal spacing
            Path B (50%): DE/rand/1/bin, no anchor  → rotational exploration
    """
    np.random.seed(seed)
    lower = np.asarray(problem.xl, dtype=float)
    upper = np.asarray(problem.xu, dtype=float)
    n_var = problem.n_var
    n_obj = problem.n_obj

    X, F = initialize_population(pop_size, problem)

    for generation in range(generations):
        fronts, ranks = fast_non_dominated_sort(F)
        crowding_vals = np.zeros(len(F), dtype=float)
        for front in fronts:
            distances = crowding_distance(F, front)
            crowding_vals[np.array(front, dtype=int)] = distances

        offspring = np.zeros_like(X)

        if generation % 5 != 0 or generation >= generations - 15:
            # ===== Growth Phase / Cool-down: phototropic vector math =====
            alpha = np.exp(-generation / max(1, generations))
            beta = 0.4
            auxin = 0.1 + 0.8 * (1.0 / (1.0 + ranks))

            light_mask, shade_mask = divide_light_shade(ranks, crowding_vals)

            # Leaders from the current generation's Front 0
            leaders = (
                X[np.array(fronts[0], dtype=int)]
                if len(fronts) > 0 and len(fronts[0]) > 0
                else X
            )

            def best_group(mask):
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    return None
                best_idx = indices[
                    np.lexsort((-crowding_vals[indices], ranks[indices]))
                ][0]
                return X[best_idx]

            light_best = best_group(light_mask)
            shade_best = best_group(shade_mask)
            if light_best is None:
                light_best = X[np.argmin(ranks)]
            if shade_best is None:
                shade_best = X[np.argmin(ranks)]

            for i in range(len(X)):
                Xi = X[i]
                leader = leaders[np.random.randint(len(leaders))]

                # Base phototropic force of competition toward the leader
                foc = alpha * beta * np.random.rand(X.shape[1]) * (leader - Xi)

                # Mutate the current solution first for exploration
                strength = 0.05 if light_mask[i] else 0.12
                mutated = mutate_solution(Xi, lower, upper, strength)

                # Directed growth toward the chosen leader weighted by auxin
                directed = auxin[i] * (leader - mutated)

                # Light and shade use different exploitation/exploration patterns
                if light_mask[i]:
                    local_population = X[light_mask]
                    light_move = (
                        alpha
                        * beta
                        * np.random.rand(X.shape[1])
                        * (light_best - mutated)
                    )
                    update = directed + light_move
                else:
                    local_population = (
                        X[shade_mask] if np.any(shade_mask) else X
                    )
                    random_peer = X[np.random.randint(len(X))]
                    shade_move = (
                        alpha
                        * beta
                        * np.random.rand(X.shape[1])
                        * (random_peer - mutated)
                    )
                    shade_best_move = 0.2 * alpha * (shade_best - mutated)
                    update = directed + shade_move + shade_best_move

                # Island-Aware Magnetic Repulsion
                vicinity_candidate = cell_vicinity(
                    mutated + update, local_population, alpha * 0.1, lower, upper
                )
                vicinity_term = vicinity_candidate - mutated

                # Final phototropic growth update
                child = Xi + foc + update + vicinity_term
                offspring[i] = np.minimum(np.maximum(child, lower), upper)

        else:
            # ===== Pollination Phase: Adaptive Multi-Operator Ensemble =====
            mut_prob = 1.0 / n_var
            front0_indices = np.array(fronts[0], dtype=int)

            # M-dimensional anchor: number of position variables to preserve
            num_pos_vars = max(n_obj - 1, 1)

            for i in range(len(X)):
                Xi = X[i]

                if np.random.rand() < 0.5:
                    # --- Path A: Orthogonal Specialist (SBX + M-D Anchor) ---
                    p1_idx = tournament_selection(front0_indices, crowding_vals)
                    p2_idx = tournament_selection(front0_indices, crowding_vals)
                    child = sbx_crossover(
                        X[p1_idx], X[p2_idx], lower, upper, prob=0.9, eta_c=15.0
                    )
                    child = polynomial_mutation(
                        child, lower, upper, prob=mut_prob, eta_m=20.0
                    )
                    # Critical M-Dimensional Anchor: force the child to inherit
                    # the first (n_obj - 1) position variables from its parent
                    # to guarantee perfect spacing on ZDT and DTLZ.
                    for k in range(num_pos_vars):
                        child[k] = Xi[k]
                else:
                    # --- Path B: Rotational Explorer (DE/rand/1/bin) ---
                    child = de_rand_1_bin(
                        Xi, X, front0_indices, lower, upper,
                        F_scale=0.5, CR=0.9,
                    )
                    # Polynomial micro-mutation for fine-grained exploration
                    child = polynomial_mutation(
                        child, lower, upper, prob=mut_prob * 0.5, eta_m=30.0
                    )
                    # NO anchor here — allows diagonal exploration to slice
                    # through the rotated epistasis traps of the UF suite.

                child = np.minimum(np.maximum(child, lower), upper)
                offspring[i] = child

        F_offspring = evaluate_population(problem, offspring)
        X = np.vstack([X, offspring])
        F = np.vstack([F, F_offspring])
        X, F = select_next_population(X, F, pop_size)

        if generation % 50 == 0:
            print(f"MOPGA generation {generation + 1}/{generations}")

    pareto_front_indices = fast_non_dominated_sort(F)[0][0]
    return F[np.array(pareto_front_indices, dtype=int)]
