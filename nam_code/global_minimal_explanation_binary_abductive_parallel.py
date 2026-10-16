import math
import os
import json
import time
import torch
import logging
import argparse
from joblib import Parallel, delayed
from helper.get_partial_models_bounds import compute_bounds


STEP = 0.0
NEGLIGIBLE_VALUE = 1e-8


def generate_vnnlib_property_full(input_value, epsilon, constant, eq, property_file, expl_indices):
    """
    Generate a vnnlib file for the verification query of a full network (f_i)
    :param input_value: input tensor to be checked
    :param epsilon: perturbation bound for the input
    :param constant: value to check if output can be greater/smaller than
    :param eq: equality operator "<" or ">"
    :param property_file: path to the vnnlib file to write
    :param expl_indices: indices of the features to be fixed in the input
    """
    input_defs = ""
    input_constraints = ""
    output_defs = ""
    output_constraints = "(assert (or\n"

    # Generate input definitions and constraints
    for i in range(len(input_value)):
        if i in expl_indices:
            input_defs += f"(declare-const X_{i} Real)\n"
            input_constraints += f"(assert (<= X_{i} {input_value[i] + 10**-30}))\n"
            input_constraints += f"(assert (>= X_{i} {input_value[i] - 10**-30}))\n"
        else:
            input_defs += f"(declare-const X_{i} Real)\n"
            input_constraints += f"(assert (<= X_{i} {input_value[i] + epsilon}))\n"
            input_constraints += f"(assert (>= X_{i} {input_value[i] - epsilon}))\n"
    
    # Generate output definition and constraints
    output_defs += f"(declare-const Y_0 Real)\n"

    # Generate output constraints for the prediction class being greater than threshold
    if eq == "<":
        output_constraints += f"\t(and (<= Y_0 {constant}))\n"
    else:
        assert eq == ">"
        output_constraints += f"\t(and (>= Y_0 {constant + NEGLIGIBLE_VALUE}))\n"
    output_constraints += "))\n"
    
    # Combine constraints into VNNLIB format
    vnnlib_property = f"; VNNLIB property for epsilon ball and winner\n\n"

    vnnlib_property += '; Definition of input variables\n'
    vnnlib_property += input_defs + "\n"

    vnnlib_property += '; Definition of output variables\n'
    vnnlib_property += output_defs + "\n"

    vnnlib_property += '; Definition of input constraints\n'
    vnnlib_property += input_constraints + "\n"

    vnnlib_property += '; Definition of output constraints\n'
    vnnlib_property += output_constraints + "\n"

    # Write property to vnnlib file
    with open(property_file, "w") as f:
        f.write(vnnlib_property)


def generate_vnnlib_property_partial(input_value, epsilon, constant, eq, property_file):
    """
    Generate a vnnlib file for the verification query of a partial network (f_i)
    :param input_value: input tensor to be checked
    :param epsilon: perturbation bound for the input
    :param constant: value to check if output can be greater/smaller than
    :param eq: equality operator "<" or ">"
    :param property_file: path to the vnnlib file to write
    """
    input_defs = ""
    input_constraints = ""
    output_defs = ""
    output_constraints = "(assert (or\n"

    # Generate input definitions and constraints
    input_defs += f"(declare-const X_0 Real)\n"
    # input_constraints += f"(assert (<= X_0 {min(flatten_sample[0] + epsilon, MAX_INPUT)}))\n"
    input_constraints += f"(assert (<= X_0 {input_value + epsilon}))\n"
    # input_constraints += f"(assert (>= X_0 {max(MIN_INPUT, flatten_sample[0] - epsilon)}))\n"
    input_constraints += f"(assert (>= X_0 {input_value - epsilon}))\n"

    # Generate output definition and constraints
    output_defs += f"(declare-const Y_0 Real)\n"

    # Generate output constraints for the prediction class being greater than threshold
    if eq == "<":
        output_constraints += f"\t(and (<= Y_0 {constant}))\n"
    else:
        assert eq == ">"
        output_constraints += f"\t(and (>= Y_0 {constant + NEGLIGIBLE_VALUE}))\n"
    output_constraints += "))\n"
    
    # Combine constraints into VNNLIB format
    vnnlib_property = f"; VNNLIB property for epsilon ball and winner\n\n"

    vnnlib_property += '; Definition of input variables\n'
    vnnlib_property += input_defs + "\n"

    vnnlib_property += '; Definition of output variables\n'
    vnnlib_property += output_defs + "\n"

    vnnlib_property += '; Definition of input constraints\n'
    vnnlib_property += input_constraints + "\n"

    vnnlib_property += '; Definition of output constraints\n'
    vnnlib_property += output_constraints + "\n"

    # Write property to vnnlib file
    with open(property_file, "w") as f:
        f.write(vnnlib_property)


def verify(
    sub_model_path, vnnlib_path, root_dir, timeout=60, cex_path="cex.txt", 
    use_subprocess=True, verifier_name='abcrown', 
    python_path="", verifier_path="", input_shape="[1,]", reason=""
):
    """
    verify a dnn verification query with alpha beta crown or marabou:
    - abcrown: generate yaml file and run with/without subprocess
    - marabou: directly run the verification query
    - extract result and counterexample
    """
    if verifier_name == 'abcrown':
        from dnnv_tools.abcrown_utils import generate_abcrown_yaml_file, solve_with_abcrown
        yaml_path = f"{root_dir}/abcrown.yaml"
        generate_abcrown_yaml_file(
            yaml_path, sub_model_path, vnnlib_path, timeout, 
            input_shape=input_shape, root_dir=root_dir
        )

        from contextlib import contextmanager

        @contextmanager
        def suppress_output():
            import sys
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        print(f'- {reason}. Running abcrown.. ', end="")
        with suppress_output():
            res, cex = solve_with_abcrown(
                yaml_path, cex_path, timeout, use_subprocess, python_path,
                verifier_path
            )
        print(f'Result: {res}, cex={cex}.')

    elif verifier_name == 'marabou':
        from dnnv_tools.marabou_utils import solve_with_marabou
        # TODO: Do we have to change the test property?
        res, cex, stats = solve_with_marabou(sub_model_path, vnnlib_path)
    else:
        raise ValueError(f"Unknown verifier name: {verifier_name}")
    return res, cex


def convert_and_save_model_to_onnx(model, path, input_size=(1, 1, 28, 28)):
    """
    Convert a PyTorch model to ONNX format and save it to the specified path.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be converted.
    - path (str): The path where the ONNX model will be saved.
    - input_size (tuple): The size of the input tensor. Default is (1, 1, 28, 28) for MNIST.
    """
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(*input_size)  # Create a dummy input with the specified size
    # print(dummy_input.shape, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},   # allow variable batch size
            'output': {0: 'batch_size'}
        },
        opset_version=13  # or newer if needed
    )

    # torch.onnx.export(
    #     model, dummy_input, path, export_params=True, opset_version=10,
    #     do_constant_folding=True, input_names=['input'], output_names=['output'],
    #     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    # )


def is_sufficient_explanation(
    f, x, S, epsilon, root_dir, timeout, cex_path, use_subprocess, 
    verifier_name, python_path, verifier_path, be_sound=True
):
    """ Check if the set S is a sufficient explanation for the model f at input x."""

    if not be_sound:
        # quick exit using sampling

        # compute correct value
        with torch.no_grad():
            y = f(x)
            # only necessary for debug
            f_i_x_i = torch.Tensor([f.feature_nns[i](x[i].unsqueeze(0)) for i in range(f.input_size)])

        # try to compute most critical value using impact factor
        x_L = x - epsilon
        x_U = x + epsilon
        # compute bounds
        with torch.no_grad():
            bounds = [
                    # only perturb features not part in the explanation
                    f_i_x_i[[i,i]]
                    if i in S else
                    torch.Tensor(compute_bounds(
                        f.feature_nns[i], x_L[i].reshape(1, -1), x_U[i].reshape(1, -1),
                        be_sound=be_sound, method=compute_bounds_method))
                    # for each feature
                    for i in range(f.input_size)
                ]
        bounds = torch.concat(bounds).reshape(-1,2)

        # multiply with additive weight
        impact_factor = bounds * f.feature_weights.unsqueeze(1)

        if y < 0:
            # maximize impact factor
            impact_factor = torch.max(impact_factor, dim=1).values
        else:
            # minimize impact factor
            impact_factor = torch.min(impact_factor, dim=1).values

        impact_factor = impact_factor - f_i_x_i * f.feature_weights.T

        # compute sum of max deviation
        y_crit_delta = torch.sum(impact_factor)

        return torch.sign(y) * y >= y_crit_delta

    # compute sound method ---

    os.makedirs(f"{root_dir}/models/onnx", exist_ok=True)
    f_onnx_path = f"{root_dir}/models/onnx/nam_full.onnx"
    convert_and_save_model_to_onnx(f, f_onnx_path, input_size=(1, *x.shape))

    os.makedirs(f"{root_dir}/properties", exist_ok=True)
    property_vnnlib_file = f"{root_dir}/properties/property_is_expl_{'-'.join(str(s) for s in S)}.vnnlib"
    # Generate vnnlib property for the sufficient explanation
    # The property is: if we fix the features of S in x, is there an input x' 
    # in [x-epsilon, x+epsilon] such that f(x') >= STEP?
    comp = "<" if f(x.reshape(1, *x.shape)) >= STEP else ">"
    generate_vnnlib_property_full(
        x, epsilon, STEP, comp, property_vnnlib_file, expl_indices=S
    )

    # Verify the property
    result, cex = verify(
        f_onnx_path, property_vnnlib_file, root_dir=root_dir,
        timeout=timeout, cex_path=cex_path, use_subprocess=use_subprocess,
        verifier_name=verifier_name, python_path=python_path,
        verifier_path=verifier_path, input_shape=(1, *x.shape),
        reason=f"Testing sufficient explanation S={S}"
    )
    # If the result is "safe", S is an explanation, otherwise it is not
    if result in ["unknown", "unsafe"]:
        print(f"Set S={S} is not a sufficient explanation.")
        return False
    elif result == "safe":
        print(f"Set S={S} is a sufficient explanation.")
        return True
    else:
        print(f"Verification result is {result} for set S={S} at input {x} and epsilon {epsilon}.")
        raise ValueError("is_sufficient_explanation(): Verification result is unknown.")

    # comp = "<" if f(x.reshape(1, *x.shape)) >= STEP else ">"
    # input_size = (1, *x.shape)
    # return is_satisfiable_comparison(
    #     f, x, epsilon, STEP, comp, "all", root_dir, timeout, cex_path, use_subprocess,
    #     verifier_name, python_path, verifier_path, input_size
    # )

def is_satisfiable_comparison(
    f_i, x_i, epsilon, M_i, compare_sign, i, root_dir, timeout, cex_path, 
    use_subprocess, verifier_name, python_path, verifier_path, input_size=(1,),
    expl_indices=[], be_sound=True, radius=math.nan
):

    # specify reason for debugging
    reason = f"i={i}: Testing f_{i}(x) {compare_sign} {M_i} (x r={radius})"

    if not be_sound:
        # quick exit using sampling (vary number of samples (1000-2000) to get different points every time)
        xs = torch.linspace(x_i-epsilon, x_i+epsilon, int(1000+torch.rand(1).item()*1000))
        with torch.no_grad():
            ys = f_i(xs)
        if compare_sign == '>':
            return "unsafe" if torch.any(ys > M_i) else "unknown", xs[torch.argmax(ys)], None
        else:
            return "unsafe" if torch.any(ys < M_i) else "unknown", xs[torch.argmin(ys)], None

    # first try to find a counterexample through sampling
    resultSample, cexSample, _ = is_satisfiable_comparison(
        f_i, x_i, epsilon, M_i, compare_sign, i, root_dir, timeout, cex_path,
        use_subprocess, verifier_name, python_path, verifier_path, input_size=input_size,
        be_sound=False, radius=radius
    )
    if resultSample == "unsafe":
        # counterexample found, no need to run verifier
        print(f'- {reason}. Running sampling.. Result: {resultSample}, cex={cexSample}')
        return resultSample, cexSample, None

    # sampling did not find a counterexample, running verifier to be sure

    # write code to generate a vnnlib file for the verification query
    # compare_sign represent if the output constraint is f_i(x_i) < M (if T=min) or f_i(x_i) > M (if T=max)
    # The query is: 
    # does there exist x' in [x_i-epsilon,x_i+epsilon], such that the output constraint is satisfiable
    # if expl_indices is not empty, the input constraints are only for the indices not in expl_indices

    # init files
    root_dir_i = f"{root_dir}_feature_{i}"
    if be_sound:
        os.makedirs(root_dir_i, exist_ok=True)
    cex_i_path = cex_path.replace(root_dir, root_dir_i).replace(".txt", f"_{i}.txt")

    os.makedirs(f"{root_dir_i}/models/onnx", exist_ok=True)
    f_i_onnx_path = f"{root_dir_i}/models/onnx/model_{i}.onnx"
    convert_and_save_model_to_onnx(f_i, f_i_onnx_path, input_size)
    
    os.makedirs(f"{root_dir_i}/properties", exist_ok=True)
    property_i_vnnlib_file = f"{root_dir_i}/properties/property_{i}.vnnlib"
    generate_vnnlib_property_partial(x_i.item(), epsilon, M_i, compare_sign, property_i_vnnlib_file)
    
    result, cex = verify(
        f_i_onnx_path, property_i_vnnlib_file, root_dir=root_dir_i,
        timeout=timeout, cex_path=cex_i_path, use_subprocess=use_subprocess,
        verifier_name=verifier_name, python_path=python_path, 
        verifier_path=verifier_path, input_shape=input_size,
        reason=reason
    )

    # check if sampled cex could be used to improve bounds
    cex2 = None
    if result == "safe" and cex is None:
        cex2 = cexSample

    return result, cex, cex2


def is_min_smaller(
    f_i, x_i, epsilon, M_i, i, root_dir, timeout, cex_path, 
    use_subprocess, verifier_name, python_path, verifier_path, be_sound=True, radius=math.nan
):
    return is_satisfiable_comparison(
        f_i, x_i, epsilon, M_i, "<", i, root_dir, timeout, cex_path, 
        use_subprocess, verifier_name, python_path, verifier_path, input_size=(1,),
        be_sound=be_sound, radius=radius
    )


def is_max_bigger(
    f_i, x_i, epsilon, M_i, i, root_dir, timeout, cex_path, 
    use_subprocess, verifier_name, python_path, verifier_path, be_sound=True, radius=math.nan
):
    return is_satisfiable_comparison(
        f_i, x_i, epsilon, M_i, ">", i, root_dir, timeout, cex_path, 
        use_subprocess, verifier_name, python_path, verifier_path, input_size=(1,),
        be_sound=be_sound, radius=radius
    )



def global_minimal_explanation_binary_abductive(
    f, x, epsilon, root_dir, timeout, cex_path, use_subprocess, verifier_name,
    python_path, verifier_path, compute_bounds_method="IBP", num_processors=4,
    sort_ratio=1.0, verifier_timeout=None, be_sound=True
):
    """
    Cardinally minimal sufficient explanation search
    f: NeuralAdditiveModel
    x: input tensor
    sort the features by v_i: the maximal change in the output of the model if the feature is removed
    when the sign of the output no longer changes when a feature is removed, the explanation is a sufficient explanation
    it is also the global minimal explanation since any other feature would have a smaller change in the output

    :param f: NeuralAdditiveModel
    :param x: input tensor
    :param epsilon: epsilon for the input bounds
    :return: S S is a cardinally minimal sufficient reason
    
    The bounds of v_i for in 1,...,n should be calculated such that F will be sorted in descending order by v_i
    They are calculated until the point that they are sortable.
    """
    start_time_part1 = time.time()
    # find T_i for each i in 1,...,n, where 
    # T_i = min if 1. f(x) > STEP and f.feature_weights > 0, or 2. min if f(x) < STEP and f.feature_weights < 0
    # T_i = max if 1. f(x) > STEP and f.feature_weights < 0, or 2. max if f(x) < STEP and f.feature_weights > 0
    T = []
    f_x = f(x.reshape(1, *x.shape))
    for i in range(f.input_size):
        with torch.no_grad():
            # f_i_x_i = f.feature_nns[i](x[i].reshape(1, ))
            if f_x > STEP and f.feature_weights[i] > 0:
                T_i = min
            elif f_x > STEP and f.feature_weights[i] < 0:
                T_i = max
            elif f_x < STEP and f.feature_weights[i] > 0:
                T_i = max
            else:  # f_x < STEP and f.feature_weights[i] < 0
                T_i = min
            T.append(T_i)
    
    # get initial bounds for f_i_x_i for each i in 1,...,n
    print('Getting initial bounds...')
    x_L = x - epsilon  # Lower bound of input
    x_U = x + epsilon  # Upper bound of input
    bounds = torch.Tensor(
        [
            compute_bounds(
                f.feature_nns[i], x_L[i].reshape(1,-1), x_U[i].reshape(1,-1), 
                be_sound=be_sound, method=compute_bounds_method)
            for i in range(f.input_size)
        ]
    )

    print('Part 1: Sorting...')
    # for i in range(f.input_size),
    #   if T[i]==min, then bounds[i,0] is the lower bound and bounds[i,1] is f_i_x_i.
    #   if T[i]==max, then bounds[i,0] is f_i_x_i and bounds[i,1] is the upper bound .
    with torch.no_grad():
        f_i_x_i = torch.Tensor([f_i(x_i.reshape(1, )) for f_i, x_i in zip(f.feature_nns, list(x))])
    bi_fi_xi = f.feature_weights * f_i_x_i
    for i in range(f.input_size):
        if T[i] == min:
            bounds[i, 1] = f_i_x_i[i]
        elif T[i] == max:
            bounds[i, 0] = f_i_x_i[i]

    unknown_i_contains_j = {}
    unknown_indices = set([])
    iter_idx = 0
    while unknown_indices != set(range(f.input_size)):
        if time.time() - start_time > timeout * sort_ratio:
            print("Timeout!")
            logging.info(f"Sorting timeout after {timeout * sort_ratio} seconds in part 1")
            
            # if v does not exits, nothing was sorted, return None
            if iter_idx == 0:
                return None, None, None, None, None, None, None, None

            # update unknown indices to include all indices with intersecting ranges
            for i in range(f.input_size):
                for j in range(f.input_size):
                    if i == j:
                        continue
                    if i in unknown_i_contains_j.get(j, set()) or \
                        j in unknown_i_contains_j.get(i, set()):
                        continue
                    li, ui = v[i]
                    lj, uj = v[j]
                    if not (li > uj or lj > ui):
                        unknown_i_contains_j.setdefault(i, set()).add(j)
                        unknown_i_contains_j.setdefault(j, set()).add(i)
                        unknown_indices.add(i)
                        unknown_indices.add(j)
            break
        # calculate 
        # v = beta_i * f_i_x_i - p =   << since p is not known exactly >>
        # beta_i * [f_i_x_i - f_i_x_i_max, f_i_x_i - f_i_x_i_min]
        # where f_i_x_i_max and f_i_x_i_min are the bounds for **the minimal value of** f_i_x_i
        v = []  # list of the values of the features in the explanation
        # each v_i has lower and upper bounds because we don't know the exact min/max of f_i_x_i:
        # if T_i == min, we bound the minimal value and subtract it from the current value:
        #   if bi is positive:
        #     the lower bound is bi_fi_xi - bi*bounds[i,1] and 
        #     the upper bound is bi_fi_xi - bi*bounds[i,0]
        #   but if pi is negative, we need to swap the bounds:
        #     the lower bound is bi_fi_xi - bi*bounds[i,0] and 
        #     the upper bound is bi_fi_xi - bi*bounds[i,1]
        # if T_i == max, we bound the maximal value and subtract the current value from it:
        #   if bi is positive:
        #     the lower bound is bounds[i,0] - bi_fi_xi and 
        #     the upper bound is bounds[i,1] - bi_fi_xi
        #   but if bi is negative, we need to swap the bounds:
        #     the lower bound is bounds[i,1] - bi_fi_xi and 
        #     the upper bound is bounds[i,0] - bi_fi_xi
        for i in range(f.input_size):
            if i in unknown_indices:
                v.append(prev_v[i])
                continue
            bi = f.feature_weights[i]
            if T[i] == min:
                if bi > 0:
                    # example: f_x=0.8>0, bi=1, Ti=min, fi_xi=0.5, bounds(on part i min value)=[0.1,0.2]
                    # v_i = [0.5-0.2, 0.5-0.1] = [0.3, 0.4]
                    # which is calculated by:
                    # bi*fi_xi[i] - bi*bounds[i,1], bi*fi_xi[i] - bi*bounds[i,0]
                    v_i = [
                        bi_fi_xi[i] - bi*bounds[i,1], bi_fi_xi[i] - bi*bounds[i,0]
                    ]
                else:  # bi < 0
                    # example: f_x=-0.8<0, bi=-1, Ti=min, fi_xi=0.5, bounds(on part i min value)=[0.1,0.2]
                    # v_i = [-0.5-(-0.1), -0.5-(-0.2)] = [-0.4, -0.3]
                    # which is calculated by:
                    # bi*fi_xi[i] - bi*bounds[i,0], bi*fi_xi[i] - bi*bounds[i,1]
                    # but we want positive v_i (the abs value of the possible change) for sorting, 
                    # so we multiple each part by -1 and swap the bounds
                    # v_i = [-0.2-(-0.5), -0.1-(-0.5)] = [0.3, 0.4]
                    v_i = [
                        bi*bounds[i,1] - bi_fi_xi[i], bi*bounds[i,0] - bi_fi_xi[i]
                    ]
            elif T[i] == max:
                if bi > 0:
                    # example: f_x=-0.8<0, bi=1, Ti=max, fi_xi=0.5, bounds(on part i max value)=[1,2] 
                    # v_i = [1-0.5, 2-0.5] = [0.5, 1.5]
                    # which is calculated by:
                    # bi*bounds[i,0] - bi_fi_xi[i], bi*bounds[i,1] - bi_fi_xi[i]
                    v_i = [
                        bi*bounds[i,0] - bi_fi_xi[i], bi*bounds[i,1] - bi_fi_xi[i]
                    ]
                else:  # bi < 0
                    # example: f_x=0.8>0, bi=-1, Ti=max, fi_xi=0.5, bounds(on part i max value)=[1,2]
                    # v_i = [-2-0.5, -1-0.5] = [-1.5, -0.5]
                    # which is calculated by:
                    # bi*bounds[i,1] - bi_fi_xi[i], bi*bounds[i,0] - bi_fi_xi[i]
                    # but we want positive v_i (the abs value of the possible change) for sorting, 
                    # so we multiple each part by -1 and swap the bounds
                    # v_i = [-0.5-(-1), -0.5-(-2)] = [0.5, 1.5]
                    v_i = [
                        bi_fi_xi[i] - bi*bounds[i,0], bi_fi_xi[i] - bi*bounds[i,1]
                    ]
            else:
                raise Exception(f"invalid T[{i}] {T[i]}")
            v.append(v_i)

        prev_v = v
        # print(f"unknown_indices={unknown_indices}")
        # print(f"v={v}")
        for j,vj in enumerate(v):
            if vj[0] < 0 or vj[1] < 0:
                print(f"Warning: v[{j}] has negative bound {vj})!")
        assert all((vi[0]>=0 and vi[1]>=0) for vi in v), f"Bounds: {v}"

        # re-compute unknown_i_contains_j after v is recomputed
        unknown_i_contains_j = {}
        for i in unknown_indices:
            unknown_i_contains_j[i] = set()
            for j in range(f.input_size):
                if j == i:
                    continue
                if v[j][0] >= v[i][0] and v[j][1] <= v[i][1]:
                    unknown_i_contains_j[i].add(j)

        # check for full order
        hasFullOrder, unordered_idx = has_full_order(v, unknown_indices, unknown_i_contains_j)
        if hasFullOrder:
            print("Found full order!")
            break

        print(f'Iteration {iter_idx}: No full order yet (Sorting: {len(unordered_idx)}/{f.input_size}). Refining {unordered_idx}..')

        # if the bounds are not sortable, we need to update them
        # calculate M = (l_i + u_i) / 2 for each i in 1,...,n
        # and check if the bounds can be updated
        M = (bounds[:,0] + bounds[:,1]) / 2
        # for i in range(f.input_size):
        #     cex_i_path = cex_path.replace(".txt", f"_{i}.txt")
        #     if T[i] == min:
        #         result, cex = is_min_smaller(
        #             f.feature_nns[i], x[i], epsilon, M[i], i, 
        #             root_dir, timeout, cex_i_path, use_subprocess, 
        #             verifier_name, python_path, verifier_path
        #         )
        #         bounds[i,1 if result == "unsafe" else 0] = M[i]
        #     elif T[i] == max:
        #         result, cex = is_max_bigger(
        #             f.feature_nns[i], x[i], epsilon, M[i], i, 
        #             root_dir, timeout, cex_i_path, use_subprocess, 
        #             verifier_name, python_path, verifier_path
        #         )
        #         bounds[i,0 if result == "unsafe" else 1] = M[i]
        #     else: raise Exception(f"invalid T[{i}] {T[i]}")
        def refine_bound(i):
            """Run one iteration of bound refinement for feature i."""
            # with open(f"/tmp/run_{i}_{jj}.txt", "w") as fw:
            #     fw.write(f"Running refinement for feature {i}\n")
            time_left = timeout * sort_ratio - (time.time() - start_time)
            if verifier_timeout is not None:
                time_left = min(time_left, verifier_timeout)
            if T[i] == min:
                #print cex into tmp file
                # if result == "unsafe":
                #     with open(f"/tmp/cex_{iter_idx}_{i}.txt", "w") as fw:
                #         fw.write(f"{iter_idx},{i},min,{M[i]},{cex}\n")
                result, cex, cex2 = is_min_smaller(
                    f.feature_nns[i], x[i], epsilon, M[i], i,
                    root_dir, time_left, cex_path, use_subprocess,
                    verifier_name, python_path, verifier_path, be_sound=be_sound, radius=(bounds[i,1]-M[i]).item()
                )

                # defaults to tested mid
                newMid = M[i]
                try:
                    # use found counterexample to improve bound
                    if not cex is None:
                        newMid = f.feature_nns[i](cex.reshape(1,1))
                except Exception as e:
                    # returned cex not always valid (None, empty, ...), keep default
                    print(e)
                    pass

                # assert result in ("safe", "unsafe"), f"min: Unexpected result: {result}"
                if result not in ["safe", "unsafe"]:
                    return (i, "unknown", result, newMid, cex2)
                else:
                    return (i, "min", result, newMid, cex2)
            elif T[i] == max:
                result, cex, cex2 = is_max_bigger(
                    f.feature_nns[i], x[i], epsilon, M[i], i,
                    root_dir, time_left, cex_path, use_subprocess,
                    verifier_name, python_path, verifier_path, be_sound=be_sound, radius=(bounds[i,1]-M[i]).item()
                )
                # defaults to tested mid
                newMid = M[i]
                try:
                    # use found counterexample to improve bound
                    newMid = f.feature_nns[i](cex.reshape(1, 1))
                except Exception as e:
                    # returned cex not always valid (None, empty, ...), keep default
                    # print(e)
                    pass

                # assert result in ("safe", "unsafe"), f"min: Unexpected result: {result}"
                if result not in ["safe", "unsafe"]:
                    return (i, "unknown", result, newMid, cex2)
                else:
                    return (i, "max", result, newMid, cex2)
            else:
                raise Exception(f"invalid T[{i}] {T[i]}")

        # run refinements in parallel across all features
        results = Parallel(n_jobs=num_processors)(delayed(refine_bound)(i)
                                     for i in unordered_idx.difference(unknown_indices))
        print('Results collected.')

        # apply updates to bounds
        tol = 1e-6
        for i, kind, result, mid, cex2 in results:
            # done with search if max/min can no longer be determined or was found
            if kind == "min":
                idx = 1 if result == "unsafe" else 0
                bounds[i, idx] = mid
                if result == "safe" and cex2 is not None:
                    y2 = f.feature_nns[i](cex2.reshape(1,1))
                    if bounds[i, 0] < y2 < bounds[i, 1]:
                        # cex2 always improves other bound
                        bounds[i, 1-idx] = y2
                    else:
                        # bound shrank (as cex2 is present) but cex2 is not in bounds
                        # -> test if bounds can be reached at all
                        print('Testing if bounds can still be reached ', end="")
                        result, _, _ = is_min_smaller(
                            f.feature_nns[i], x[i], epsilon, bounds[i,0]+tol, i,
                            root_dir, 20, cex_path, use_subprocess,
                            verifier_name, python_path, verifier_path, be_sound=be_sound,
                            radius=(bounds[i, 1] - M[i]).item()
                        )
                        if result == "safe":
                            # found bounds can no longer be reached.
                            print(f' -> Bounds for feature {i} can no longer be reached.')
                            bounds[i,1] = bounds[i,0] # +tol

            elif kind == "max":
                idx = 0 if result == "unsafe" else 1
                bounds[i, idx] = mid
                if result == "safe" and cex2 is not None:
                    y2 = f.feature_nns[i](cex2.reshape(1,1))
                    if bounds[i, 0] < y2 < bounds[i, 1]:
                        # cex2 always improves other bound
                        bounds[i, 1-idx] = y2
                    else:
                        # bound shrank (as cex2 is present) but cex2 is not in bounds
                        # -> test if bounds can be reached at all
                        print('Testing if bounds can still be reached ', end="")
                        result, _, _ = is_max_bigger(
                            f.feature_nns[i], x[i], epsilon, bounds[i,1]-tol, i,
                            root_dir, 20, cex_path, use_subprocess,
                            verifier_name, python_path, verifier_path, be_sound=be_sound,
                            radius=(bounds[i, 1] - M[i]).item()
                        )
                        if result == "safe":
                            # found bounds can no longer be reached.
                            print(f' -> Bounds for feature {i} can no longer be reached.')
                            bounds[i,0] = bounds[i,1] # -tol

            if not be_sound:
                # check if overshot initial unsound bounds
                if bounds[i, 1] - bounds[i, 0] < 0:
                    # correct with newly found min/max
                    bounds[i, 0] = mid
                    bounds[i, 1] = mid

            # check unknown or where bounds converged (2*epsilon for numeric stability)
            if kind == "unknown":
                unknown_indices.add(i)
                # check if any other feature range in v is contained in v_i's range
                # in this case, we can't sort them soundly, and the sort might be incorrect in 1 element

        iter_idx += 1
            
        # import numpy as np
        # np.savetxt(f'/tmp/bounds_parallel_iter_{iter_idx}.csv', bounds, delimiter=',', fmt='%f')
        

    # while not has_full_order(bounds):
    #     M = (bounds[:,0] + bounds[:,1]) / 2
    #     for i in range(f.input_size):
    #         cex_i_path = cex_path.replace(".txt", f"_{i}.txt")
    #         if T[i] == min:
    #             result, cex = is_min_smaller(
    #                 f.feature_nns[i], x[i], epsilon, M[i], i, 
    #                 root_dir, timeout, cex_i_path, use_subprocess, 
    #                 verifier_name, python_path, verifier_path
    #             )
    #             bounds[i,1 if result == "unsafe" else 0] = M[i]
    #         elif T[i] == max:
    #             result, cex = is_max_bigger(
    #                 f.feature_nns[i], x[i], epsilon, M[i], i, 
    #                 root_dir, timeout, cex_i_path, use_subprocess, 
    #                 verifier_name, python_path, verifier_path
    #             )
    #             bounds[i,0 if result == "unsafe" else 1] = M[i]
    #         else: raise Exception(f"invalid T[{i}] {T[i]}")
            
    #     # calculate p = min(beta_i * f_i_x_i) if f(x) > 0, else max(beta_i * f_i_x_i)
    #     p = []
    #     for i in range(f.input_size):
    #         if f_x >= STEP and f.feature_weights[i] > 0:
    #             p_i = f.feature_weights[i] * bounds[i,0]
    #         elif f_x >= STEP and f.feature_weights[i] < 0:
    #             p_i = f.feature_weights[i] * bounds[i,1]
    #         elif f_x < STEP and f.feature_weights[i] > 0:
    #             p_i = f.feature_weights[i] * bounds[i,1]
    #         else:
    #             p_i = f.feature_weights[i] * bounds[i,0]
    #         p.append(p_i.item())
    #     p = torch.Tensor(p)
    #     print(f"p1={sorted(p)}")

    #     # # another way to get p. for some reason it has different p and hence different results
    #     # lower_bounds = f.feature_weights * bounds[:,0]#.reshape(1,-1)
    #     # upper_bounds = f.feature_weights * bounds[:,1]#.reshape(1,-1)
    #     # lower_bounds = lower_bounds.reshape(1,*lower_bounds.shape)
    #     # upper_bounds = upper_bounds.reshape(1,*upper_bounds.shape)
    #     # concat_bounds = torch.cat((lower_bounds, upper_bounds), axis=0)
    #     # # if T[i] == min:
    #     # if f_x >= 0:
    #     #     p2 = concat_bounds.min(axis=0).values
    #     # elif f_x < 0: #T[i] == max:
    #     #     p2 = concat_bounds.max(axis=0).values
    #     # else:
    #     #     raise Exception(f"invalid f_x] {f_x}")
    #     # print(f"p2={sorted(p2)}")

    #     # calculate v = beta_i * f_i_x_i - p
    #     with torch.no_grad():
    #         f_i_x_i = torch.Tensor([f_i(x_i.reshape(1, )) for f_i, x_i in zip(f.feature_nns, list(x))])
    #     v = f.feature_weights * f_i_x_i - p
    #     assert (v>0).all()

    total_time_part1 = time.time() - start_time_part1

    print('Part 2: Searching...')

    start_time_part2 = time.time()
    # F <- {1, ..., n} Features for iteration
    F = list(range(f.input_size))
        
    # Sort F in descending order by the values of v. v includes intervals, but
    # after the loop ends, these intervals has no intersection, and one value 
    # can be used for sorting, e.g. the maximal value of the interval.
    
    # print(f"v={v}")
    # print(f"unknown_indices={unknown_indices}")
    
    # we sort by max(v[i]), so:
    # it sort features that are not in unknown_i_contains_j correctly.
    # if i is in unknown_i_contains_j:
    #   - if j in unknown_indices[i], i is before j, and this couple might be not soundly sorted
    #   - if j not in unknown_indices[i], this couple is soundly sorted
    F.sort(key=lambda i: max(v[i]), reverse=True)
    
    # return F
    print(f"F={F}")
    # end of Alg. 2 in the paper

    # Alg. 4 implementation
    # binary search of the first index m in F such that F[:m] is sufficient explanation.
    UB = len(F)  # upper bound of the search
    LB = 0  # lower bound of the search
    step = 0
    per_feature_pairs = []  # (index_in_F, dt)
    feature_times = [0] * len(F)  # time for each feature in F to be included in the explanation
    while UB != LB:
        m = (UB + LB) // 2
        # check if F[:m] is a sufficient explanation
        S = set(F[:m])
        # if verbose:
        #     print(f"[binary] binsearch step={step} try_size={m} LB={LB} UB={UB}")
        t_step = time.time()
        time_left = timeout - (time.time() - start_time_part1)
        if verifier_timeout is not None:
            time_left = min(time_left, verifier_timeout)
        if time_left <= 0:
            logging.info(f"Searching timeout after {time.time() - start_time_part2} seconds in part 2")
            break

        sufficient = is_sufficient_explanation(
            f, x, S, epsilon, root_dir, time_left, cex_path, use_subprocess, 
            verifier_name, python_path, verifier_path, be_sound=be_sound
        )
        dt = time.time() - t_step
        if sufficient:
            feature_times[LB:m] = [feature_times[i] + dt for i in range(LB, m)]
            # Assign time to boundary index m-1 (if any). Features from m..UB-1 are freed at once; do not double count.
            if m - 1 >= 0:
                per_feature_pairs.append((m - 1, dt))
            UB = m
        else:
            feature_times[m:UB] = [feature_times[i] + dt for i in range(m, UB)]
            if m < len(F):
                per_feature_pairs.append((m, dt))
            LB = m + 1
        step += 1
        
    if time_left <= 0:
        total_time_part2 = timeout - total_time_part1
    else:
        total_time_part2 = time.time() - start_time_part2
    return F[:UB], total_time_part1, total_time_part2, unknown_i_contains_j, F, per_feature_pairs, feature_times, v, bounds, f_x

    # # S <- {} The current sufficient reason
    # S = set()

    # # for each i in F do
    # for i in F:
    #     # check if the sign can be changed by adding i to S
    #     # sign_1 should be the sign of the sum of two groups: 1. beta_i * f_i(x_i) for i in S and 2. p_j + beta_0 for j in S'
    #     new_result = sum([f.feature_weights[j]*f_i_x_i[j] for j in S]) + sum([p[j] for j in set(F).difference(S)]) + f.bias
    #     sign_1 = 1 if new_result >= STEP else -1
    #     # sign_2 is just the sign of the original result
    #     sign_2 = 1 if f_x >= STEP else -1
    #     if sign_1 * sign_2 >= 0:
    #         break
    #     # S <- S union {i}
    #     S.add(i)
    
    # # return S S is a cardinally minimal sufficient reason
    # return S


def has_full_order(v_bounds, unknown_indices, unknown_i_contains_j):
    """
    Check if there is a full order on the variables, meaning that for every pair (li, ui) and (lj, uj),
    either li > uj or lj > ui.
    
    :param bounds: List of tuples [(l1, u1), (l2, u2), ...], the values of the 
                   minimal and maximal changes in the output when each feature 
                   is removed.
    :param unknown_i_contains_j: maps from unknown indices to set of indices, 
                                 each one's range is contained in the unknown index range.
    :return: True if a full order exists, False otherwise

    # Examples of usage
    print(has_full_order([(1, 3), (4, 6), (7, 9)]))  # Output: True
    print(has_full_order([(1, 5), (4, 6), (7, 9)]))  # Output: False
    print(has_full_order([(4, 6), (1, 5), (7, 9)]))  # Output: False
    # with unknown_i_contains_j
    print(has_full_order([(4, 6), (1, 7), (8, 9)], {1: set([0])}))  # Output: True
    """
    n = len(v_bounds)
    unordered_idx = set()
    for i in range(n):
        for j in range(i + 1, n):
            # if i is unknown, and i range contains j range, skip (i,j)
            if i in unknown_i_contains_j and j in unknown_i_contains_j[i]:
                continue
            li, ui = v_bounds[i]
            lj, uj = v_bounds[j]
            if not (li > uj or lj > ui):
                # Overlapping or unordered variables found
                if not li == ui:  # not converged
                    unordered_idx.add(i)
                if not lj == uj:  # not converged
                    unordered_idx.add(j)

    # filter unknown indices
    unordered_idx = unordered_idx.difference(unknown_indices)

    return len(unordered_idx) == 0, unordered_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Script for managing YAML generation and verification")
    # Arguments
    parser.add_argument('--root_dir', type=str, default="./abcrown_dir",
                        help="Root directory for the project.")
    parser.add_argument('--dataset', type=str, default="breast_cancer",
                        help="Name of the dataset.")
    parser.add_argument('--sample_index', type=int, default=1,
                        help="Index of sample in dataset.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Size of each batch from the dataset (default: 1).")
    parser.add_argument('--num_processors', type=int, default=8,
                        help="Number of processors to use for parallel feature sorting")
    parser.add_argument('--is_bigger_nam', default=False, action='store_true',
                        help="Flag to indicate if using the bigger NAM model.")
    parser.add_argument('--network_path', type=str, default="models/pth/nam_full.pth",
                        help="Path to the neural network model.")
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help="Perturbation bound for verification.")
    parser.add_argument('--timeout', type=int, default=600,
                        help="Timeout in seconds for each verification query.")
    parser.add_argument('--sort_ratio', type=float, default=0.8,
                        help="Ratio of sorting timeout with respect to the overall timeout.")
    parser.add_argument('--verifier_timeout', type=float, default=45.0,
                        help="Timeout for each verification query.")
    parser.add_argument('--use_subprocess', default=False,
                        help="Flag to use subprocess for execution.")
    parser.add_argument('--cex_path', type=str, default="./cex.txt",
                        help="Path to the file contains the counter-example.")
    parser.add_argument('--verifier_path', type=str, default="./abcrown_dir/complete_verifier/abcrown.py",
                        help="Path to the verifier script.")
    parser.add_argument('--python_path', type=str, default="",
                        help="Path to the python interpreter.")
    parser.add_argument('--verbose', default=False,
                        help="Flag to enable verbose mode.")
    parser.add_argument('--verifier_name', type=str, default="abcrown",
                        help="Name of the verifier to use.")
    parser.add_argument('--compute_bounds_method', type=str, default="IBP",
                        help="Name of the compute bounds method (in Auto-LiRPA) to use.")
    parser.add_argument('--device', type=str, default="cpu",
                        help="device type to use (cpu/gpu).")
    parser.add_argument('--exp_log_dir', type=str, default="exp_logs_3_credit_parallel",
                        help="Relative path to the directory for experiment logs.")
    parser.add_argument('--be_sound', default=True,
                        help="Flag to switch between verifying an sampling.")
    return parser.parse_args()


def print_args(args):
    # print all args
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    timeout = args.timeout
    root_dir = args.root_dir
    cex_path = args.cex_path
    use_subprocess = args.use_subprocess
    verifier_name = args.verifier_name
    python_path = args.python_path
    verifier_path = args.verifier_path
    device = args.device
    sample_index = args.sample_index
    compute_bounds_method = args.compute_bounds_method
    dataset = args.dataset
    num_processors = args.num_processors
    # network_path = args.network_path
    network_path = f"models/{dataset}/nam_full.pth"
    epsilon = args.epsilon
    batch_size = args.batch_size
    is_bigger_nam = args.is_bigger_nam
    sort_ratio = args.sort_ratio
    verifier_timeout = args.verifier_timeout
    be_sound = args.be_sound

    # exp_log_dir = "exp_logs" # old format (without sorting & searching times)
    # exp_log_dir = "exp_logs_2_heloc"  # new format (with sorting & searching times)
    exp_log_dir = args.exp_log_dir

    os.makedirs(exp_log_dir, exist_ok=True)
    log_file = f"{exp_log_dir}/run__{dataset}__{sample_index}__{epsilon}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        filemode="w",  # overwrite each run; change to "a" to append
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Arguments: {args}")

    # load the full model
    from helper.nam_train_test_save_load import load_full_model, test_model, load_data
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)
    loaded_model = load_full_model(network_path, input_size, device, is_bigger_nam)
    test_model(loaded_model, test_loader, device)

    # load the input
    X_batch, y_batch = list(test_loader)[sample_index]  # change to [sample_index] if batch_size == 1
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    x = X_batch[0]

    # if y_batch[0] == 1:
    #     exit(-1)

    result = {
        'dataset': dataset,
        'instance_id': int(sample_index),
        'epsilon': float(epsilon),
        'timeout': int(timeout),
        'device': device,
        'cpus_cores': int(num_processors),
        'feat_sort_method': 'ours',
        'finished': False,
        'time': {
            # 'startup': float(startup_time), --> sorting_time
            'per_feature': []  # searching time
        },
    }

    # get the explanation
    try:
        start_time = time.time()
        explanation, part1_time, part2_time, unknown_indices, F_full, \
            per_feature_pairs, feature_times, impact_bounds, partial_net_bounds, f_x = \
            global_minimal_explanation_binary_abductive(
            loaded_model, x, epsilon, root_dir, timeout, cex_path, use_subprocess, 
            verifier_name, python_path, verifier_path, compute_bounds_method, num_processors,
            sort_ratio, verifier_timeout, be_sound
        )
        end_time = time.time()

        # Build per_feature times aligned with feat_order (F_full)
        per_feature = None
        if F_full is not None:
            per_feature = [0.0 for _ in range(len(F_full))]
            for idx, dt in per_feature_pairs:
                if 0 <= idx < len(per_feature):
                    per_feature[idx] += float(dt)
            is_part = [False for _ in range(len(F_full))]
            k = len(explanation)
            for i in range(k):
                is_part[i] = True
            result.update({
                'feat_order': F_full,
                'is_part_explanation': is_part,
                'impact_bounds': [[float(ib[0]),float(ib[1])] for ib in impact_bounds],
                'partial_net_bounds': [[pnb[0].item(), pnb[1].item()] for pnb in partial_net_bounds]
            })
        if part1_time is not None:
            result['time'].update({
                'sorting_time': float(part1_time),
                'per_feature': per_feature,
                'full_feature_times': feature_times
            })
        result['finished'] = part1_time is not None and part2_time is not None

        # perhaps we could cache this in the method above, but we wouldnt save much
        result['f_x_i'] = list([loaded_model.feature_nns[i](xi.reshape(-1)).float().item() for (i, xi) in enumerate(x)])
        result['f_x'] = f_x.float().item()

        print(f"\n\n{result} \n\n")
        
        summary = ",\n ".join([
                f"dataset={dataset}",
                f"sample_index={sample_index}",
                f"epsilon={epsilon}",
                f"length/total={len(explanation) if explanation is not None else 'N/A'}/{input_size}",
                f"Time taken: {end_time - start_time:.2f} seconds",
                f"Sorting time={part1_time if part1_time is not None else 'N/A'} seconds",
                f"Searching time={part2_time if part2_time is not None else 'N/A'} seconds",
                f"global minimal explanation={list(explanation) if explanation is not None else 'N/A'}",
                f"unknown_indices={list(unknown_indices.items()) if unknown_indices is not None else 'N/A'}",
                f"full_feature_times={feature_times}",
                f"per_feature_times={per_feature}",
                f"feat_order={F_full}",
                f"impact_bounds={impact_bounds if impact_bounds is not None else 'N/A'}",
                f"partial_net_bounds={partial_net_bounds if partial_net_bounds is not None else 'N/A'}"
            ])
        logging.info(summary)
        print(summary)
        with open(f"{exp_log_dir}/result__{dataset}__{sample_index}__{epsilon}.json", "w") as fw:
            json.dump(result, fw, indent=4)
        print(f"Result saved to {exp_log_dir}/result__{dataset}__{sample_index}__{epsilon}.json")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
        print(traceback.format_exc())
