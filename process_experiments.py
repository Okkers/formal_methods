import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sufficient_explanations import load_results, brute_force, find_all_subsets
import time

def frequencies(tuples, n):
    counts = [0] * n
    for t in tuples:
        for x in t:
            counts[x] += 1

    # print(counts)
    total = sum(counts)
    if total == 0:
        return np.array(counts)
    return np.array(counts)


def plot_frequencies(counts, len_subsets, k, processed_path, i):
    os.makedirs(f"{processed_path}/frequencies", exist_ok=True)

    non_zero = counts[counts > 0]
    normalize = non_zero.min() if len(non_zero) > 0 else 1
    
    print("SMALLEST: ", normalize)
    print("BIGGEST: ", counts.max())

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(counts)), counts / normalize)
    plt.xlabel("Feature")
    plt.ylabel("Frequency / smallest non-zero frequency")
    plt.yscale("log")
    plt.title(f"Feature frequencies for instance {i} (subsets={len_subsets}, k={k})")
    plt.xticks(range(len(counts)))
    plt.tight_layout()
    plt.savefig(f"{processed_path}/frequencies/frequencies_{i}.png")
    plt.close()


if __name__ == "__main__":
    folder = "breast_cancer_30_eps_40"
    path = f"results/{folder}"

    processed_path = f"results_processed/{folder}"
    os.makedirs(processed_path, exist_ok=True)

    with open(f"{path}/parameters.json") as params:
        params = json.load(params)

    N = params["N"]
    epsilon = params["epsilon"]


    RUN_WITH_BRUTE = False

    runtime_brute = []
    runtime_fast = []

    logs = []
    for i in range(N):
        log = dict()
        # logs.push[log]

        res = load_results(f"{path}/result__breast_cancer__{i}__{epsilon}.json")

        f_x = res['f_x']

        # impact bounds are not sorted, in the json they i-th bound corresponds to i-th feature
        # we sort them in the order of the json and we keep the feature mask to keep track what is where
        impact_bounds = [res["impact_bounds"][i] for i in res["feat_order"]]
        feature_mask = res["feat_order"]

        ls = np.array([b[0] for b in impact_bounds])
        k = int(np.array(res['is_part_explanation']).sum())
        log['k'] = k

        subsets = None
        if RUN_WITH_BRUTE:
            t1 = time.time()
            subsets = brute_force(f_x, ls, k)
            runtime_brute.append(time.time() - t1)
            print(f"{len(subsets)} cardinally minimal subsets were found with k={k}")
            brute_sets = {tuple(sorted(s)) for s in subsets}
            brute_counts = frequencies(brute_sets, len(ls))
            log['runtime_brute'] = runtime_brute[-1]

        t2 = time.time()
        fast_subsets = find_all_subsets(f_x, ls, k)
        runtime_fast.append(time.time() - t2)
        log['runtime_fast'] = runtime_fast[-1]

        print(f"{len(fast_subsets)} subsets found by find_all_subsets with k={k}")

        fast_sets = {tuple(sorted(indices)) for indices, _ in fast_subsets}

        if RUN_WITH_BRUTE:
            print(f"Both methods agree: {brute_sets == fast_sets}")
  
        fast_counts = frequencies(fast_sets, len(ls))[feature_mask]

        if subsets:
            # Index 0 is unused because feature IDs are 1-based.
            features_in_all_subsets = fast_counts == len(subsets)
        else:
            features_in_all_subsets = fast_counts == len(fast_sets)

        in_intersection = np.where(features_in_all_subsets)[0].tolist()
        intersection = int(features_in_all_subsets.sum())

        print(f"There are {intersection} features that are in all sets")
        log['intersection_size'] = intersection
        log['features_in_intersection'] = in_intersection
        logs.append(log)

        plot_frequencies(fast_counts, len(fast_sets), k, processed_path, i)
        print("\n")

        # print("Runtimes Brute:", runtime_brute[-1])
        # print("-----")
        # print("Runtimes Fast:", runtime_fast[-1])
        
        os.makedirs(f"{processed_path}/individual", exist_ok=True)
        with open(f'{processed_path}/individual/results_{i}.json', 'w') as f:
            json.dump(log, f, indent=4)
            f.write("\n")



# ================Summary===================
# ===========================================
    def summarize(values):
        if not values:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    summary = {
        "k": summarize([log["k"] for log in logs]),
    }
    summary['n_data_points'] = N
    summary['epsilon'] = epsilon

    if runtime_brute:
        summary["runtime_brute"] = summarize(runtime_brute)

    if runtime_fast:
        summary["runtime_fast"] = summarize(runtime_fast)

    # Intersections
    tuples = [log['features_in_intersection'] for log in logs]
    freq = frequencies(tuples, len(ls))
    summary['counts'] = freq.tolist()

    with open(f"{processed_path}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")



