from generate_data import generate_dataset, generate_sample
from sufficient_explanations import brute_force, find_all_subsets


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
