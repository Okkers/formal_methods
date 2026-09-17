import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- helpers ---
def parse_log(filepath):
    """Return (time_taken, explanation_size) or None if timeout/error."""
    time_taken = None
    explanation_size = None
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # check timeout / error
    if any("Timeout" in line or "- ERROR -" in line for line in lines):
        return None
    
    for line in lines:
        # match time
        m_time = re.search(r"Time taken: ([0-9.]+) seconds", line)
        if m_time:
            time_taken = float(m_time.group(1))
        
        # match explanation
        m_exp1 = re.search(r"global minimal explanation=\[(.*?)\]", line)
        m_exp2 = re.search(r"algorithm1 minimal explanation=\[(.*?)\]", line)
        
        if m_exp1:
            items = m_exp1.group(1).strip()
            explanation_size = 0 if items == "" else len(items.split(","))
        elif m_exp2:
            items = m_exp2.group(1).strip()
            explanation_size = 0 if items == "" else len(items.split(","))
    
    if time_taken is None or explanation_size is None:
        return None
    return time_taken, explanation_size


def collect_results(directory):
    results = []
    for fname in os.listdir(directory):
        if fname.endswith(".log"):
            parsed = parse_log(os.path.join(directory, fname))
            if parsed:
                results.append(parsed)  # (time, size)
    return results


def compute_curve(results, thresholds):
    """Compute average explanation size up to threshold T."""
    curve = []
    for T in thresholds:
        filtered = [size for (time, size) in results if time <= T]
        avg_size = np.mean(filtered) if filtered else np.nan
        curve.append(avg_size)
    return curve


# --- main ---
dataset = "heloc"
dir1 = f"./results/exp_logs_4_{dataset}_parallel_8"
dir2 = f"./results/new_exp_logs_{dataset}_local_minima_naive"

results1 = collect_results(dir1)
results2 = collect_results(dir2)

all_times = [t for (t, _) in results1 + results2]
thresholds = np.linspace(0, max(all_times), 50)

curve1 = compute_curve(results1, thresholds)
curve2 = compute_curve(results2, thresholds)

plt.figure(figsize=(7, 5))
plt.plot(thresholds, curve1, label="Ours", marker="o")
plt.plot(thresholds, curve2, label="Local Minima (Naive)", marker="s")
plt.xlabel("Time threshold T (seconds)")
plt.ylabel("Average explanation size (≤ T)")
plt.title(f"Average Explanation Size vs Time Threshold ({dataset})")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(f"avg_exp_size_per_timeout_comparison_{dataset}.png")