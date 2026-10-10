from itertools import combinations
import numpy as np

import json

def load_results(path):
    with open(path) as result_file:
        return json.load(result_file)

from itertools import combinations

def find_all_subsets(f_x, deltas, k):
    n = len(deltas)
    m = n - k
    threshold = abs(f_x)
    deltas = list(deltas)

    order = sorted(range(n), key=lambda i: deltas[i])
    sorted_vals = [deltas[i] for i in order]

    prefix_sum = [0.0] * (n + 1)
    for j in range(n):
        prefix_sum[j + 1] = prefix_sum[j] + sorted_vals[j]

    results = []
    chosen = []  

    def best_case_completion(start, slots_needed):
        end = start + slots_needed
        if end > n:
            return float("inf")
        return prefix_sum[end] - prefix_sum[start]

    def backtrack(start, partial_sum):
        slots_needed = m - len(chosen)

        if slots_needed == 0:
            if partial_sum < threshold:
                complement_idx = {order[j] for j in chosen}
                explanation_idx = sorted(set(range(n)) - complement_idx)
                results.append((explanation_idx, [deltas[i] for i in explanation_idx]))
            return

        if n - start < slots_needed:
            return

        if not (partial_sum + best_case_completion(start, slots_needed)) < threshold:
            return

        chosen.append(start)
        backtrack(start + 1, partial_sum + sorted_vals[start])
        chosen.pop()

        backtrack(start + 1, partial_sum)

    backtrack(0, 0.0)
    return results

def brute_force(f_x, deltas, k):
    n = len(deltas)
    threshold = abs(f_x)

    matches = []
    for S in combinations(range(n), k):
        S_set = set(S)
        comp_sum = sum(deltas[i] for i in range(n) if i not in S_set)
        if comp_sum < threshold:   # <- this was the problem all along :')
            matches.append(S)
    return matches



if __name__ == "__main__":
    # path = "exp_logs_3_credit_parallel/result__breast_cancer__1__0.4.json"
    # path = "exp_logs_3_credit_parallel/result__breast_cancer__1__0.15.json"
    path = "exp_logs_3_credit_parallel/result__breast_cancer__2__0.3.json"
    # path = "exp_logs_3_credit_parallel/result__credit__0__0.3.json"
    res = load_results(f"nam_code/{path}")

    f_x = res['f_x']

    # impact bounds are not sorted, in the json they i-th bound corresponds to i-th feature
    # we sort them in the order of the json and we keep the feature mask to keep track what is where
    impact_bounds = [res["impact_bounds"][i] for i in res["feat_order"]]
    feature_mask = res["feat_order"]

    ls = np.array([b[0] for b in impact_bounds])
    k = np.array(res['is_part_explanation']).sum()

    print(f"f_x = {f_x}")
    print(k)
    print(ls)

    print(ls[:-k].sum())
    # print(ls.sum() - ls[0] - ls[19] - ls[24])

    subsets = brute_force(f_x, ls, k)

    print(f"{len(subsets)} cardinally minimal subsets were found")