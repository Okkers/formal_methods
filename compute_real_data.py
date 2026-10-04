from __future__ import annotations
import json
from find_subsets import find_all_subsets

def load_real_sample_from_json(json_path):
    with open(json_path) as fh:
        result = json.load(fh)

    feat_order = result["feat_order"]
    is_part = result["is_part_explanation"]
    impact_bounds = result["impact_bounds"]
    f_x_i = result["f_x_i"]
    f_x = result["f_x"]

    n = len(feat_order)
    k = sum(1 for flag in is_part if flag)

    deltas = [0.0] * n
    for pos, orig_idx in enumerate(feat_order):
        li, ui = impact_bounds[pos]
        fi_x = f_x_i[orig_idx]
        deltas[orig_idx] = max(fi_x - li, fi_x - ui)

    official_explanation = sorted(feat_order[pos] for pos, flag in enumerate(is_part) if flag)

    return {
        "f_x": f_x,
        "deltas": deltas,
        "k": k,
        "feat_order": feat_order,
        "official_explanation": official_explanation,
        "raw": result,
    }

def find_all_explanations_for_real_sample(json_path, verify = True):

    sample = load_real_sample_from_json(json_path)
    explanations = find_all_subsets(sample["f_x"], sample["deltas"], sample["k"])

    if verify:
        found_sets = {tuple(idx) for idx, _ in explanations}
        official = tuple(sample["official_explanation"])
        if official not in found_sets:
            print("we missed some sets.")

    return sample, explanations

if __name__ == "__main__":
    pth_to_data = "nam_code/exp_logs_3_credit_parallel/result__breast_cancer__1__0.15.json"

    test = find_all_explanations_for_real_sample(pth_to_data)

    print(test)
