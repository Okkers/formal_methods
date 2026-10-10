import numpy as np 

def generate_sample(n, eps, seed= None, slope_range = (-5.0, 5.0), input_range = (-1.0, 1.0)):
    rng = np.random.default_rng(seed)
    a = rng.uniform(*slope_range, size=n)

    x = rng.uniform(*input_range, size=n)
    beta0 = rng.uniform(-1.0, 1.0)
    f_x = beta0 + float(np.sum(a * x))
    deltas = np.abs(a) * eps

    return {
        "a": a,
        "x": x,
        "beta0": beta0,
        "f_x": f_x,
        "eps": eps,
        "deltas": deltas,
    }

def generate_dataset(num_samples, n, eps, seed= None):
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(num_samples):
        sample = generate_sample(n=n, eps=eps, seed=int(rng.integers(0, 10**6)))
        dataset.append(sample)
    return dataset