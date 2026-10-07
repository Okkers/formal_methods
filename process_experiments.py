import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sufficient_explanations import load_results, brute_force

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
    plt.savefig(f"{processed_path}/frequencies_{i}.png")
    plt.close()


if __name__ == "__main__":
    folder = "breast_cancer_5_eps_15"
    path = f"results/{folder}"

    processed_path = f"results_processed/{folder}"
    os.makedirs(processed_path, exist_ok=True)

    with open(f"{path}/parameters.json") as params:
        params = json.load(params)

    N = params["N"]
    epsilon = params["epsilon"]


    for i in range(N):
        res = load_results(f"{path}/result__breast_cancer__{i}__{epsilon}.json")

        f_x = res['f_x']

        # impact bounds are not sorted, in the json they i-th bound corresponds to i-th feature
        # we sort them in the order of the json and we keep the feature mask to keep track what is where
        impact_bounds = [res["impact_bounds"][i] for i in res["feat_order"]]
        feature_mask = res["feat_order"]

        ls = np.array([b[0] for b in impact_bounds])
        k = np.array(res['is_part_explanation']).sum()

        subsets = brute_force(f_x, ls, k)
        # print(subsets)
        
        print(f"{len(subsets)} cardinally minimal subsets were found with k={k}")
        counts = frequencies(subsets, len(ls))

        if subsets:
            # Index 0 is unused because feature IDs are 1-based.
            features_in_all_subsets = counts == len(subsets)
        else:
            features_in_all_subsets = np.zeros(len(ls), dtype=bool)

        print(f"There are {features_in_all_subsets.sum()} features that are in all sets")
        plot_frequencies(counts, len(subsets), k, processed_path, i)
        print("\n")


    


