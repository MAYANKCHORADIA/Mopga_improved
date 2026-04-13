import numpy as np


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
    """Move a solution toward the local cell vicinity of the population."""
    if len(population) == 0:
        return x

    sample_size = max(1, min(len(population), len(population) // 5))
    neighbors = population[np.random.choice(len(population), sample_size, replace=True)]
    direction = np.mean(neighbors, axis=0) - x
    candidate = x + radius * direction
    return np.minimum(np.maximum(candidate, lower), upper)


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


def run_mopga(problem, pop_size, generations, seed=1):
    """Run the Multi-Objective Phototropic Growth Algorithm and return Pareto front objectives."""
    np.random.seed(seed)
    lower = np.asarray(problem.xl, dtype=float)
    upper = np.asarray(problem.xu, dtype=float)

    X, F = initialize_population(pop_size, problem)

    for generation in range(generations):
        fronts, ranks = fast_non_dominated_sort(F)
        crowding = np.zeros(len(F), dtype=float)
        for front in fronts:
            distances = crowding_distance(F, front)
            crowding[np.array(front, dtype=int)] = distances

        # Decaying phototropic coefficient over generations
        alpha = np.exp(-generation / max(1, generations))
        beta = 0.4

        # Auxin is stronger for better-ranked individuals (rank 0 is best)
        auxin = 0.1 + 0.8 * (1.0 / (1.0 + ranks))

        light_mask, shade_mask = divide_light_shade(ranks, crowding)

        # Pareto leaders come from the best nondominated front
        leaders = X[np.array(fronts[0], dtype=int)] if len(fronts) > 0 and len(fronts[0]) > 0 else X

        def best_group(mask):
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return None
            best_idx = indices[np.lexsort((-crowding[indices], ranks[indices]))][0]
            return X[best_idx]

        light_best = best_group(light_mask)
        shade_best = best_group(shade_mask)
        if light_best is None:
            light_best = X[np.argmin(ranks)]
        if shade_best is None:
            shade_best = X[np.argmin(ranks)]

        offspring = np.zeros_like(X)
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
                light_move = alpha * beta * np.random.rand(X.shape[1]) * (light_best - mutated)
                update = directed + light_move
            else:
                local_population = X[shade_mask] if np.any(shade_mask) else X
                random_peer = X[np.random.randint(len(X))]
                shade_move = alpha * beta * np.random.rand(X.shape[1]) * (random_peer - mutated)
                shade_best_move = 0.2 * alpha * (shade_best - mutated)
                update = directed + shade_move + shade_best_move

            # Local vicinity exploration around the mutated solution
            vicinity_candidate = cell_vicinity(mutated + update, local_population, alpha * 0.1, lower, upper)
            vicinity_term = vicinity_candidate - mutated

            # Final phototropic growth update
            child = Xi + foc + update + vicinity_term
            offspring[i] = np.minimum(np.maximum(child, lower), upper)

        F_offspring = evaluate_population(problem, offspring)
        X = np.vstack([X, offspring])
        F = np.vstack([F, F_offspring])
        X, F = select_next_population(X, F, pop_size)

        if generation % 20 == 0:
            print(f"MOPGA generation {generation + 1}/{generations}")

    pareto_front_indices = fast_non_dominated_sort(F)[0][0]
    return F[np.array(pareto_front_indices, dtype=int)]
