from itertools import combinations
import numpy as np

import json

def load_results(path):
    with open(path) as result_file:
        return json.load(result_file)

def find_all_subsets(f_x, ls, k):
    '''
    Finds all subsets of size k whose complement contains only values greater
    than abs(f_x).

    Returns pairs of (subset values, subset indices).
    '''
    threshold = abs(f_x)
    subsets = []

    return subsets

if __name__ == "__main__":
    path = "exp_logs_3_credit_parallel/result__breast_cancer__1__0.4.json"
    path = "exp_logs_3_credit_parallel/result__credit__0__0.3.json"
    res = load_results(f"nam_code/{path}")

    f_x = res['f_x']

    # impact bounds are not sorted, in the json they i-th bound corresponds to i-th feature
    # we sort them in the order of the json and we keep the feature mask to keep track what is where
    impact_bounds = [res["impact_bounds"][i] for i in res["feat_order"]]
    feature_mask = res["feat_order"]

    ls = np.array([b[0] for b in impact_bounds])
    k = np.array(res['is_part_explanation']).sum()

    subsets = find_all_subsets(f_x, ls, k)

    print(subsets)