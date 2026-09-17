import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
exp_dir = "exp_logs_1"
out_dir = "analysis_results_1"
os.makedirs(out_dir, exist_ok=True)

epsilons = [0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
sample_indices = range(20)

# regex patterns
args_pattern = re.compile(r"Arguments: Namespace\(.*dataset='([^']+)', sample_index=(\d+).*epsilon=([\d.]+).*")
result_pattern = re.compile(r"dataset=([^,]+), sample_index=(\d+), epsilon=([\d.]+), length/total=(\d+)/(\d+), Time taken: ([\d.]+) seconds, global minimal explanation=\[.*\]")

# storage
times = defaultdict(list)            # (epsilon) -> list of times
times_all = defaultdict(list)        # includes unfinished
sizes = defaultdict(list)            # (epsilon) -> list of explanation sizes
sizes_all = defaultdict(list)
finished_per_sample = defaultdict(int)
finished_per_epsilon = defaultdict(int)

max_time = 0
total_input_sizes = {}  # per (sample, epsilon) the total input size

# --- Parse logs ---
for fname in os.listdir(exp_dir):
    if not fname.startswith("run__breast_cancer__"):
        continue
    fpath = os.path.join(exp_dir, fname)
    with open(fpath, "r") as f:
        content = f.read()

    # check if result line exists
    result_match = result_pattern.search(content)
    args_match = args_pattern.search(content)

    if args_match:
        dataset, sample_idx, epsilon = args_match.groups()
        sample_idx = int(sample_idx)
        epsilon = float(epsilon)

    if result_match:
        dataset, sample_idx, epsilon, exp_len, total_len, t = result_match.groups()
        sample_idx = int(sample_idx)
        epsilon = float(epsilon)
        exp_len = int(exp_len)
        total_len = int(total_len)
        t = float(t)
        max_time = max(max_time, t)

        times[epsilon].append(t)
        times_all[epsilon].append(t)
        sizes[epsilon].append(exp_len)
        sizes_all[epsilon].append(exp_len)
        finished_per_sample[sample_idx] += 1
        finished_per_epsilon[epsilon] += 1
        total_input_sizes[(sample_idx, epsilon)] = total_len
    else:
        # unfinished
        if args_match:
            total_len = None
            # fallback: estimate total input size if we have any finished runs with same epsilon
            for k, v in total_input_sizes.items():
                if abs(k[1] - epsilon) < 1e-8:
                    total_len = v
                    break
            if total_len is None:
                total_len = 0  # safe default

            times_all[epsilon].append(max_time + 1.0)  # mark unfinished as slightly worse
            sizes_all[epsilon].append(total_len)

# --- Analysis / Plots ---

# 1) average time per epsilon
def plot_avg_time(times_dict, title, fname):
    avgs = [np.mean(times_dict[e]) if e in times_dict and len(times_dict[e]) > 0 else 0 for e in epsilons]
    plt.figure()
    plt.plot(epsilons, avgs, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("average time (s)")
    plt.title(title)
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

plot_avg_time(times, "Avg Time per Epsilon (finished only)", "avg_time_finished.png")
plot_avg_time(times_all, "Avg Time per Epsilon (all, unfinished penalized)", "avg_time_all.png")

# 2) average explanation size per epsilon
def plot_avg_size(size_dict, title, fname):
    avgs = [np.mean(size_dict[e]) if e in size_dict and len(size_dict[e]) > 0 else 0 for e in epsilons]
    plt.figure()
    plt.plot(epsilons, avgs, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("average explanation size")
    plt.title(title)
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

plot_avg_size(sizes, "Avg Explanation Size per Epsilon (finished only)", "avg_size_finished.png")
plot_avg_size(sizes_all, "Avg Explanation Size per Epsilon (all)", "avg_size_all.png")

# 3) bar plot of number of finished experiments per sample
plt.figure()
samples = list(range(20))
vals = [finished_per_sample[s] for s in samples]
plt.bar(samples, vals)
plt.xlabel("Sample Index")
plt.ylabel("Finished Experiments")
plt.title("Finished Experiments per Sample")
plt.savefig(os.path.join(out_dir, "finished_per_sample.png"))
plt.close()

# 4) bar plot of number of finished experiments per epsilon
plt.figure()
vals = [finished_per_epsilon[e] for e in epsilons]
plt.bar([str(e) for e in epsilons], vals)
plt.xlabel("epsilon")
plt.ylabel("Finished Experiments")
plt.title("Finished Experiments per Epsilon")
plt.savefig(os.path.join(out_dir, "finished_per_epsilon.png"))
plt.close()

print(f"Analysis complete. Results saved in: {out_dir}")
