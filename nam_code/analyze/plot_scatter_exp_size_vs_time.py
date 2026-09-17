import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_log_with_dataset(filepath):
    """Return (dataset, time, explanation_size) or None if timeout/error."""
    fname = os.path.basename(filepath)
    parts = fname.split("__")
    if len(parts) < 4:
        return None
    dataset = parts[1]

    with open(filepath, "r") as f:
        lines = f.readlines()

    # skip timeouts/errors
    if any("Timeout" in line or "- ERROR -" in line for line in lines):
        return None

    time_taken = None
    explanation_size = None
    for line in lines:
        # time
        m_time = re.search(r"Time taken: ([0-9.]+) seconds", line)
        if m_time:
            time_taken = float(m_time.group(1))

        # explanation size
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
    return dataset, time_taken, explanation_size


def collect_all(directory):
    results = []
    for fname in os.listdir(directory):
        if fname.endswith(".log"):
            parsed = parse_log_with_dataset(os.path.join(directory, fname))
            if parsed:
                results.append(parsed)  # (dataset, time, size)
    return results


# --- main ---
dataset = "breast_cancer"
dir1 = f"./results/exp_logs_4_{dataset}_parallel_8"
dir2 = f"./results/new_exp_logs_{dataset}_local_minima_sensitive"

res1 = collect_all(dir1)
res2 = collect_all(dir2)

datasets = sorted(set(d for d, _, _ in res1 + res2))
colors = {ds: c for ds, c in zip(datasets, plt.cm.tab10.colors)}

plt.figure(figsize=(7, 5))

# scatter for method 1
for d, t, s in res1:
    plt.scatter(s, t, color=colors[d], marker="o", label=f"{d}: Ours")

# scatter for method 2
for d, t, s in res2:
    plt.scatter(s, t, color=colors[d], marker="x", label=f"{d}: Local Minima (Naive)")

plt.xlabel("Explanation size")
plt.ylabel("Time taken (seconds)")
plt.title("Scatter: Explanation size vs Time per dataset")
# avoid duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(f"scatter_exp_size_vs_time_{dataset}.png")
