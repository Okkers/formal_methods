from generate_data import generate_dataset, generate_sample
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

def brute_force(f_x, deltas):
    n = len(deltas)
    threshold = abs(f_x)
    fits = (lambda total: total < threshold)

    for k in range(n + 1):
        matches = []
        for S in combinations(range(n), k):
            complement = [i for i in range(n) if i not in S]
            comp_sum = sum(deltas[i] for i in complement)
            if fits(comp_sum):
                matches.append(S)
        if matches:
            return k, matches
    return n, [tuple(range(n))]


def compare_brute_force(sample):
    f_x, deltas = sample["f_x"], sample["deltas"]

    k_bf, matches_bf = brute_force(f_x, deltas)
    bf_sets = {tuple(sorted(S)) for S in matches_bf}

    fast_results = find_all_subsets(f_x, deltas, k_bf)
    fast_sets = {tuple(idx) for idx, _ in fast_results}

    match = bf_sets == fast_sets

    if not match:
        print("k:", k_bf)
        print("brute force no.:", len(bf_sets))
        print("find_all_subsets:", len(fast_sets))


        print("diff: ", {bf_sets - fast_sets})
        print("diff: ", {fast_sets - bf_sets})

    return match, fast_sets

if __name__ == "__main__":
    print("single sample:")

    sample = generate_sample(n=8, eps=0.1, seed=0)
    ok = compare_brute_force(sample)

    print("GOOD" if ok else "BAD")

    print("dataset of 20 samples:")
    ds = generate_dataset(num_samples=20, n=10, eps=0.15, seed=42)
    mismatches = 0
    for i, s in enumerate(ds):

        ya, fs = compare_brute_force(s)
        if not ya:
            mismatches += 1
            print(f"sample {i} broken")

        else:
            print("sample")
            print(fs)
            print("-----")
            

    print("no. of mismatches: ", mismatches)
