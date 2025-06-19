from terminating_instance import *

import matplotlib.pyplot as plt
import os
import time

def run_algorithm_until_termination(n, alpha, V=None, W=None):
    if V is None or W is None:
        V, W = generate_random_matrices(n)

    mu = compute_mu(V, W)
    optimal_M_L, _ = compute_max_weight_matching(V, W, n)
    sw_opt = compute_social_welfare(V, W, optimal_M_L)

    M_L, M_R = compute_max_weight_matching(V, W, n)

    iterations = 0
    while True:
        blocking_pairs = find_blocking_pairs(V, W, M_L, M_R, n, alpha)
        if not blocking_pairs:
            break
        i, j = random.choice(blocking_pairs)
        old_r, old_l = M_L.pop(f"M{i}", None), M_R.pop(f"W{j}", None)
        M_L[f"M{i}"] = f"W{j}"
        M_R[f"W{j}"] = f"M{i}"
        if old_r and old_r in M_R:
            del M_R[old_r]
        if old_l and old_l in M_L:
            del M_L[old_l]
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
        "iterations": iterations,
        "mu": mu,
        "sw_final": sw_final,
        "sw_opt": sw_opt,
        "ratio": ratio,
        "threshold": threshold,
        "guarantee_met": meets_threshold
    }

def run_experiments_avg_iterations(num_trials=100, n=10, alpha=0.9):
    total_iterations = 0
    max_iterations = 0
    for _ in range(num_trials):
        result = run_algorithm_until_termination(n, alpha)
        iters = result["iterations"]
        total_iterations += iters
        if iters > max_iterations:
            max_iterations = iters
    average_iterations = total_iterations / num_trials
    return average_iterations, max_iterations

def get_plot_filename(name=None, kind="avg", folder="plots"):
    os.makedirs(folder, exist_ok=True)
    if name:
        return os.path.join(folder, f"{name}_{kind}.png")
    else:
        prefix = f"{kind}_plot"
        existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".png")]
        nums = [int(f[len(prefix):-4]) for f in existing if f[len(prefix):-4].isdigit()]
        next_num = max(nums) + 1 if nums else 1
        return os.path.join(folder, f"{prefix}{next_num}.png")

# Configuration
start_time = time.time()

alphas = [0.75, 0.8, 0.85, 0.9, 0.95]
num_trials = 200
n_range = list(range(10, 21))

from matplotlib.cm import get_cmap
cmap = get_cmap('tab10')

# Store results
results = {alpha: {"avg": [], "max": []} for alpha in alphas}

# Run experiments
for alpha in alphas:
    print(f"\nRunning for alpha = {alpha}")
    for n in n_range:
        avg, max_iter = run_experiments_avg_iterations(num_trials=num_trials, n=n, alpha=alpha)
        results[alpha]["avg"].append(avg)
        results[alpha]["max"].append(max_iter)

#custom_name = input("Enter plot name (leave empty to auto-generate): ").strip()

custom_name = None

# Generate filenames based on user input
avg_plot_path = get_plot_filename(name=custom_name if custom_name else None, kind="avg")
max_plot_path = get_plot_filename(name=custom_name if custom_name else None, kind="max")

# Plot average iterations
plt.figure(figsize=(10, 6))
for idx, alpha in enumerate(alphas):
    plt.plot(n_range, results[alpha]["avg"], label=f"α = {alpha}", color=cmap(idx))
plt.xlabel("n (number of agents)")
plt.ylabel("Average iterations to termination")
plt.title("Average Iterations vs. n")
plt.legend()
plt.tight_layout()
plt.savefig(avg_plot_path)
plt.show()

# Plot max iterations
plt.figure(figsize=(10, 6))
for idx, alpha in enumerate(alphas):
    plt.plot(n_range, results[alpha]["max"], label=f"α = {alpha}", color=cmap(idx))
plt.xlabel("n (number of agents)")
plt.ylabel("Max iterations to termination")
plt.title("Max Iterations vs. n")
plt.legend()
plt.tight_layout()
plt.savefig(max_plot_path)
plt.show()

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"\nExecution time: {elapsed_minutes:.2f} minutes")

print(f"Saved average plot to {avg_plot_path}")
print(f"Saved max plot to {max_plot_path}")