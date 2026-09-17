import os
import torch
from joblib import Parallel, delayed
from .vnnlib_utils import generate_vnnlib_property_partial, generate_vnnlib_property_full
from .onnx_utils import convert_and_save_model_to_onnx
from .verify_utils import verify

STEP = 0.0

# Aliases for clarity when passing around kinds
MIN, MAX = min, max

def is_satisfiable_comparison(f_i, x_i, epsilon, M_i, compare_sign, i, root_dir, timeout,
                              cex_path, use_subprocess, verifier_name, python_path, verifier_path,
                              input_size=(1,), expl_indices=[]):
    os.makedirs(f"{root_dir}/models/onnx", exist_ok=True)
    f_i_onnx_path = f"{root_dir}/models/onnx/model_{i}.onnx"
    convert_and_save_model_to_onnx(f_i, f_i_onnx_path, input_size)

    os.makedirs(f"{root_dir}/properties", exist_ok=True)
    property_i_vnnlib_file = f"{root_dir}/properties/property_{i}.vnnlib"
    generate_vnnlib_property_partial(x_i.item(), epsilon, M_i, compare_sign, property_i_vnnlib_file)

    result, cex = verify(
        f_i_onnx_path, property_i_vnnlib_file, root_dir=root_dir,
        timeout=timeout, cex_path=cex_path, use_subprocess=use_subprocess,
        verifier_name=verifier_name, python_path=python_path,
        verifier_path=verifier_path, input_shape=input_size
    )
    return result, cex


def is_min_smaller(*args, **kwargs):
    args = list(args)
    # compare_sign "<"
    return is_satisfiable_comparison(*args, "<", *args[4:])  # not used; kept for API parity


def is_max_bigger(*args, **kwargs):
    args = list(args)
    return is_satisfiable_comparison(*args, ">", *args[4:])  # not used; kept for API parity


def is_sufficient_explanation(f, x, S, epsilon, root_dir, timeout, cex_path, use_subprocess,
                              verifier_name, python_path, verifier_path):
    os.makedirs(f"{root_dir}/models/onnx", exist_ok=True)
    f_onnx_path = f"{root_dir}/models/onnx/nam_full.onnx"
    convert_and_save_model_to_onnx(f, f_onnx_path, input_size=(1, *x.shape))

    os.makedirs(f"{root_dir}/properties", exist_ok=True)
    property_vnnlib_file = f"{root_dir}/properties/property_is_expl_{'-'.join(str(s) for s in S)}.vnnlib"
    comp = "<" if f(x.reshape(1, *x.shape)) >= STEP else ">"
    generate_vnnlib_property_full(x, epsilon, STEP, comp, property_vnnlib_file, expl_indices=S)

    result, cex = verify(
        f_onnx_path, property_vnnlib_file, root_dir=root_dir,
        timeout=timeout, cex_path=cex_path, use_subprocess=use_subprocess,
        verifier_name=verifier_name, python_path=python_path,
        verifier_path=verifier_path, input_shape=(1, *x.shape)
    )
    if result == "unsafe":
        return False
    elif result == "safe":
        return True
    else:
        raise ValueError("is_sufficient_explanation(): Verification result is unknown.")


def has_full_order(bounds, unknown_i_contains_j):
    n = len(bounds)
    for i in range(n):
        for j in range(i + 1, n):
            if i in unknown_i_contains_j and j in unknown_i_contains_j[i]:
                continue
            li, ui = bounds[i]
            lj, uj = bounds[j]
            if not (li > uj or lj > ui):
                return False
    return True


def refine_bounds_until_order(
    f, x, epsilon, bounds, T_kinds, mode, root_dir, timeout, cex_path,
    use_subprocess, verifier_name, python_path, verifier_path, active=None, num_processors=4, verbose=False
):
    """Generic binary-search refinement of per-feature bounds until they are strictly ordered.

    - f: NAM with feature_nns and feature_weights
    - bounds: Tensor [n,2] of initial (lo, hi) per feature
    - T_kinds: list of MIN or MAX indicating which side to tighten for each feature
    - mode: 'parallel' or 'serial'
    Returns the refined bounds tensor.
    """
    input_size = f.input_size
    if active is None:
        active = list(range(input_size))

    def _refine_once(i, kind, mid, root_dir_i):
        cex_i_path = cex_path.replace(".txt", f"_{i}.txt")
        if kind == MIN:
            result, _ = is_satisfiable_comparison(
                f.feature_nns[i], x[i], epsilon, mid, "<", i,
                root_dir_i, timeout, cex_i_path, use_subprocess, verifier_name, python_path, verifier_path, input_size=(1,)
            )
            return ("min", result, mid)
        elif kind == MAX:
            result, _ = is_satisfiable_comparison(
                f.feature_nns[i], x[i], epsilon, mid, ">", i,
                root_dir_i, timeout, cex_i_path, use_subprocess, verifier_name, python_path, verifier_path, input_size=(1,)
            )
            return ("max", result, mid)
        else:
            raise ValueError(f"invalid T kind for feature {i}: {kind}")

    # Loop until full order achieved
    iter_idx = 0
    while not has_full_order(bounds, {}):
        M = (bounds[:, 0] + bounds[:, 1]) / 2
        if verbose:
            lo = bounds[:,0].min().item(); hi = bounds[:,1].max().item()
            print(f"[refine] iter={iter_idx} active={len(active)} mid_range=({M.min().item():.4f},{M.max().item():.4f}) bounds_span=({lo:.4f},{hi:.4f}) mode={mode}")
        if mode == "parallel":
            def job(i):
                root_dir_i = f"{root_dir}_feature_{i}"
                os.makedirs(root_dir_i, exist_ok=True)
                return (i, *_refine_once(i, T_kinds[i], M[i], root_dir_i))
            results = Parallel(n_jobs=num_processors)(delayed(job)(i) for i in active)
        else:
            results = []
            for i in active:
                results.append((i, *_refine_once(i, T_kinds[i], M[i], root_dir)))

        # Update bounds per result
        cnt_safe = cnt_unsafe = cnt_unknown = 0
        for i, kind, result, mid in results:
            if result not in ("safe","unsafe"):
                cnt_unknown += 1
                continue
            if result == "unsafe":
                cnt_unsafe += 1
            else:
                cnt_safe += 1
            if kind == "min":
                # if UNSAFE: exists x' s.t. f_i(x') < mid  -> upper bound becomes mid
                # if SAFE:   cannot go below mid -> lower bound becomes mid
                bounds[i, 1 if result == "unsafe" else 0] = mid
            elif kind == "max":
                # if UNSAFE: exists x' s.t. f_i(x') > mid  -> lower bound becomes mid
                # if SAFE:   cannot exceed mid -> upper bound becomes mid
                bounds[i, 0 if result == "unsafe" else 1] = mid
        if verbose:
            print(f"[refine] iter={iter_idx} results: safe={cnt_safe} unsafe={cnt_unsafe} unknown={cnt_unknown}")
        iter_idx += 1
    if verbose:
        print("[refine] achieved full order.")
    return bounds