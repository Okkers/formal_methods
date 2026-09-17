import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory with logs

# breast_cancer:
# log_dir = "exp_logs_2"
# analysis_dir = "analysis_results_2"
# heloc, last version:
log_dir = "exp_logs_3_heloc"
analysis_dir = "analysis_results_3_heloc"
os.makedirs(analysis_dir, exist_ok=True)

# Regex patterns
args_pattern = re.compile(r"dataset=(\w+), sample_index=(\d+), epsilon=([\d.]+)")
result_pattern = re.compile(
    r"length/total=(\d+)/(\d+), Time taken: ([\d.]+) seconds, Sorting time=([\d.]+) seconds, Searching time=([\d.]+) seconds, global minimal explanation=\[([0-9, ]*)\]"
)

# Data storage
records = []

# Parse all log files
for fname in os.listdir(log_dir):
    if not fname.endswith(".log"):
        continue
    with open(os.path.join(log_dir, fname)) as f:
        lines = f.readlines()

    # skip empty files
    if len(lines) == 0:
        continue

    # Extract dataset, sample_index, epsilon from filename
    parts = fname.replace(".log", "").split("__")
    if len(parts) < 4:
        continue
    dataset, sample_index, epsilon = parts[1], int(parts[2]), float(parts[3])

    # Default: not finished
    finished = False
    explanation_size = None
    total_inputs = None
    total_time = None
    sorting_time = None
    searching_time = None
    explanation = None

    # If there is a results line, parse it
    if len(lines) > 1:
        match = result_pattern.search(lines[1])
        if match:
            finished = True
            explanation_size = int(match.group(1))
            total_inputs = int(match.group(2))
            total_time = float(match.group(3))
            sorting_time = float(match.group(4))
            searching_time = float(match.group(5))
            explanation = [int(x.strip()) for x in match.group(6).split(",") if x.strip().isdigit()]

    records.append({
        "dataset": dataset,
        "sample_index": sample_index,
        "epsilon": epsilon,
        "finished": finished,
        "explanation_size": explanation_size,
        "total_inputs": total_inputs,
        "total_time": total_time,
        "sorting_time": sorting_time,
        "searching_time": searching_time,
        "explanation": explanation
    })

# Convert to DataFrame
df = pd.DataFrame(records)

# -------------------------------
# 1) Average time per epsilon
# -------------------------------
time_stats = df[df["finished"]].groupby("epsilon")["total_time"].agg(["mean", "std"])
time_stats.plot(y="mean", yerr="std", kind="bar", legend=False)
plt.ylabel("Average Time (s)")
plt.title("Average Time per Epsilon (Finished Only)")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "avg_time_per_epsilon.png"))
plt.close()

# -------------------------------
# 2) Average explanation size per epsilon
# -------------------------------
size_stats = df[df["finished"]].groupby("epsilon")["explanation_size"].agg(["mean", "std"])
size_stats.plot(y="mean", yerr="std", kind="bar", legend=False)
plt.ylabel("Average Explanation Size")
plt.title("Average Explanation Size per Epsilon (Finished Only)")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "avg_explanation_size_per_epsilon.png"))
plt.close()

# -------------------------------
# 3) Scatter plot: total/sorting/searching time per epsilon
# -------------------------------
import matplotlib.pyplot as plt

# Use only finished experiments
finished_df = df[df["finished"]].copy()

# Group by epsilon and average the times
time_components = finished_df.groupby("epsilon")[["sorting_time", "searching_time"]].mean()
time_components["other_time"] = finished_df.groupby("epsilon")["total_time"].mean() - time_components.sum(axis=1)

# Create stacked bar plot
time_components[["sorting_time", "searching_time", "other_time"]].plot(
    kind="bar",
    stacked=True,
    figsize=(10,6)
)

plt.ylabel("Average Time (s)")
plt.xlabel("Epsilon")
plt.title("Average Time Breakdown per Epsilon")
plt.legend(title="Time Component")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "bar_time_breakdown_per_epsilon.png"))
plt.close()

# finished_df = df[df["finished"]]
# plt.scatter(finished_df["epsilon"], finished_df["total_time"], label="Total Time", alpha=0.7)
# plt.scatter(finished_df["epsilon"], finished_df["sorting_time"], label="Sorting Time", alpha=0.7)
# plt.scatter(finished_df["epsilon"], finished_df["searching_time"], label="Searching Time", alpha=0.7)
# plt.xlabel("Epsilon")
# plt.ylabel("Time (s)")
# plt.title("Times per Epsilon")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(analysis_dir, "scatter_times_per_epsilon.png"))
# plt.close()

# -------------------------------
# 4) Bar plot: finished vs not finished per sample_index
# -------------------------------
sample_status = df.groupby(["sample_index", "finished"]).size().unstack(fill_value=0)
sample_status.plot(kind="bar", stacked=True)
plt.ylabel("Number of Experiments")
plt.title("Finished vs Not-Finished per Sample Index")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "finished_vs_not_per_sample.png"))
plt.close()

# -------------------------------
# 5) Bar plot: finished vs not finished per epsilon
# -------------------------------
epsilon_status = df.groupby(["epsilon", "finished"]).size().unstack(fill_value=0)
epsilon_status.plot(kind="bar", stacked=True)
plt.ylabel("Number of Experiments")
plt.title("Finished vs Not-Finished per Epsilon")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "finished_vs_not_per_epsilon.png"))
plt.close()

print(f"Analysis complete. Results saved in {analysis_dir}")
