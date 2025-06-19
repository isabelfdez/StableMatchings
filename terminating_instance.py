import numpy as np
import networkx as nx
import random

def generate_random_matrices(n, min_val=0, max_val=100):
    V = np.random.randint(min_val, max_val + 1, (n, n))
    W = np.random.randint(min_val, max_val + 1, (n, n))
    for i in range(n):
        for j in range(n):
            if V[i, j] == 0:
                W[j, i] = 0
            if W[j, i] == 0:
                V[i, j] = 0
    return V, W

def compute_max_weight_matching(V, W, n):
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            weight = V[i, j] + W[j, i]
            G.add_edge(f"M{i}", f"W{j}", weight=weight)
    matching = nx.max_weight_matching(G, maxcardinality=True)
    M_L, M_R = {}, {}
    for m, w in matching:
        if "M" in m:
            M_L[m], M_R[w] = w, m
        else:
            M_L[w], M_R[m] = m, w
    return M_L, M_R

def find_blocking_pairs(V, W, M_L, M_R, n, alpha):
    blocking_pairs = []
    for i in range(n):
        current_partner = M_L.get(f"M{i}", None)
        for j in range(n):
            if current_partner == f"W{j}":
                continue
            v_curr = V[i, int(current_partner[1:])] if current_partner else -1
            v_new = V[i, j]
            w_curr = W[j, int(M_R[f"W{j}"][1:])] if f"W{j}" in M_R else -1
            w_new = W[j, i]
            if alpha * v_new > v_curr and alpha * w_new > w_curr:
                blocking_pairs.append((i, j))
    return blocking_pairs

def compute_mu(V, W):
    mu_candidates = []
    n = V.shape[0]
    for i in range(n):
        for j in range(n):
            vij = V[i, j]
            wji = W[j, i]
            if vij > 0 and wji > 0:
                ratio1 = vij / wji
                ratio2 = wji / vij
                mu_candidates.append(min(ratio1, ratio2))
    return min(mu_candidates, default=1.0)

def compute_social_welfare(V, W, matching):
    total = 0
    for m, w in matching.items():
        i = int(m[1:])
        j = int(w[1:])
        total += V[i, j] + W[j, i]
    return total

def run_algorithm(n, alpha, max_iterations=100, V=None, W=None):
    if V is None or W is None:
        V, W = generate_random_matrices(n)

    mu = compute_mu(V, W)
    optimal_M_L, _ = compute_max_weight_matching(V, W, n)
    sw_opt = compute_social_welfare(V, W, optimal_M_L)

    M_L, M_R = compute_max_weight_matching(V, W, n)

    iterations = 0
    while iterations < max_iterations:
        blocking_pairs = find_blocking_pairs(V, W, M_L, M_R, n, alpha)
        if not blocking_pairs:
            break
        i, j = random.choice(blocking_pairs)
        old_w, old_m = M_L.pop(f"M{i}", None), M_R.pop(f"W{j}", None)
        M_L[f"M{i}"] = f"W{j}"
        M_R[f"W{j}"] = f"M{i}"
        if old_w and old_w in M_R:
            del M_R[old_w]
        if old_m and old_m in M_L:
            del M_L[old_m]
        matched_pairs = compute_max_weight_matching(V, W, n)
        for lagent, ragent in matched_pairs[0].items():
            if lagent not in M_L and ragent not in M_R:
                M_L[lagent] = ragent
                M_R[ragent] = lagent
        iterations += 1

    sw_final = compute_social_welfare(V, W, M_L)
    threshold = (1 / alpha) * (mu / (mu + 1))
    ratio = sw_final / sw_opt if sw_opt > 0 else 0
    meets_threshold = ratio >= threshold

    return {
        "terminated": iterations < max_iterations,
        "mu": mu,
        "sw_final": sw_final,
        "sw_opt": sw_opt,
        "ratio": ratio,
        "threshold": threshold,
        "guarantee_met": meets_threshold
    }

def run_experiments(num_trials=100, n=10, alpha=0.9, max_iterations=100):
    terminated_count = 0
    guarantee_met_count = 0
    for _ in range(num_trials):
        result = run_algorithm(n, alpha, max_iterations)
        if result["terminated"]:
            terminated_count += 1
        if result["guarantee_met"]:
            guarantee_met_count += 1
    print(f"n = {n}")
    print(f"→ Terminated in {terminated_count} / {num_trials}")
    print(f"→ Approximation guarantee met in {guarantee_met_count} / {num_trials}")

# Example experiment loop
if __name__ == "__main__":
    for n in range(10, 21):
        run_experiments(num_trials=20, n=n, alpha=0.9, max_iterations=20)