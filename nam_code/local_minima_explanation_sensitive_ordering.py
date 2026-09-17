import os
import time
import torch
import logging
import argparse


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


def verify(
    sub_model_path, vnnlib_path, root_dir, timeout=60, cex_path="cex.txt", 
    use_subprocess=True, verifier_name='abcrown', 
    python_path="", verifier_path="", input_shape="[1,]"
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
            yaml_path, sub_model_path, vnnlib_path, timeout, input_shape=input_shape
        )
        res, cex = solve_with_abcrown(
            yaml_path, cex_path, timeout, use_subprocess, python_path, 
            verifier_path
        )
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
    verifier_name, python_path, verifier_path
):
    """ Check if the set S is a sufficient explanation for the model f at input x."""

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
        verifier_path=verifier_path, input_shape=(1, *x.shape)
    )
    # If the result is "unsafe", S is not an explanation, otherwise it is
    if result == "unsafe":
        print(f"Set S={S} is not a sufficient explanation for the model at input {x}.")
        return False
    elif result == "safe":
        print(f"Set S={S} is a sufficient explanation for the model at input {x}.")
        return True
    else:
        print(f"Verification result is unknown for set S={S} at input {x}.")
        raise ValueError("is_sufficient_explanation(): Verification result is unknown.")

    # comp = "<" if f(x.reshape(1, *x.shape)) >= STEP else ">"
    # input_size = (1, *x.shape)
    # return is_satisfiable_comparison(
    #     f, x, epsilon, STEP, comp, "all", root_dir, timeout, cex_path, use_subprocess,
    #     verifier_name, python_path, verifier_path, input_size
    # )


def parse_args():
    parser = argparse.ArgumentParser(description="Script for managing YAML generation and verification")
    # Arguments
    parser.add_argument('--root_dir', type=str, default="./abcrown_dir",
                        help="Root directory for the project.")
    parser.add_argument('--dataset', type=str, default="breast_cancer",  # breast_cancer
                        help="Name of the dataset.")
    parser.add_argument('--sample_index', type=int, default=1,
                        help="Index of sample in dataset.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Size of each batch from the dataset (default: 1).")
    parser.add_argument('--is_bigger_nam', default=False, action='store_true',
                        help="Flag to indicate if using the bigger NAM model.")
    parser.add_argument('--network_path', type=str, default="models/pth/nam_full.pth",
                        help="Path to the neural network model.")
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help="Perturbation bound for verification.")
    parser.add_argument('--timeout', type=int, default=5,
                        help="Timeout in seconds for each verification query (default: one minute).")
    parser.add_argument('--use_subprocess', default=False,
                        help="Flag to use subprocess for execution.")
    parser.add_argument('--cex_path', type=str, default="/home/yizhak/Research/Code/GlobalExplanationForAdditiveModels/cex.txt",
                        help="Path to the file contains the counter-example.")
    parser.add_argument('--verifier_path', type=str, default="/home/yizhak/Research/Code/alpha-beta-CROWN/complete_verifier/abcrown.py",
                        help="Path to the verifier script.")
    parser.add_argument('--python_path', type=str, default="/home/yizhak/miniconda3/envs/ab-crown/bin/python",
                        help="Path to the python interpreter.")
    parser.add_argument('--verbose', default=False,
                        help="Flag to enable verbose mode.")
    parser.add_argument('--verifier_name', type=str, default="abcrown",
                        help="Name of the verifier to use.")
    parser.add_argument('--compute_bounds_method', type=str, default="IBP",
                        help="Name of the compute bounds method (in Auto-LiRPA) to use.")
    parser.add_argument('--device', type=str, default="cpu",
                        help="device type to use (cpu/gpu).")
    parser.add_argument('--feature_ordering', type=str, default="sensitivity",
                        help="Method to use for feature ordering (naive/sensitivity).")
    parser.add_argument('--exp_log_dir', type=str, default="exp_logs_3_credit_parallel",
                        help="Relative path to the directory for experiment logs.")
    return parser.parse_args()


def print_args(args):
    # print all args
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def order_features_by_impact(f, x, perturbation=1e-6):
    """
    Order features by their impact on the model output.
    Returns features ordered from least impactful to most impactful.
    
    :param f: NeuralAdditiveModel
    :param x: input tensor
    :param perturbation: small perturbation value (default: 1e-6)
    :return: list of feature indices ordered by impact (least to most impactful)
    """
    print(f"Ordering features by impact using perturbation={perturbation}")
    
    # Get original prediction
    with torch.no_grad():
        original_output = f(x.reshape(1, *x.shape))
        print(f"Original output: {original_output.item():.6f}")
    
    feature_impacts = []
    
    # Calculate impact for each feature
    for i in range(f.input_size):
        # Create perturbed input
        x_perturbed = x.clone()
        x_perturbed[i] = x_perturbed[i] + perturbation
        
        # Get perturbed prediction
        with torch.no_grad():
            perturbed_output = f(x_perturbed.reshape(1, *x_perturbed.shape))
        
        # Calculate absolute difference (impact)
        impact = abs(perturbed_output.item() - original_output.item())
        feature_impacts.append((i, impact))
        
        print(f"Feature {i}: original_val={x[i].item():.6f}, "
              f"perturbed_val={x_perturbed[i].item():.6f}, "
              f"output_change={perturbed_output.item() - original_output.item():.8f}, "
              f"absolute_impact={impact:.8f}")
    
    # Sort features by impact (ascending: least to most impactful)
    feature_impacts.sort(key=lambda x: x[1])
    
    # Extract feature indices in order
    ordered_features = [feature_idx for feature_idx, impact in feature_impacts]
    
    print(f"Features ordered by impact (least to most): {ordered_features}")
    print(f"Corresponding impacts: {[impact for _, impact in feature_impacts]}")
    
    return ordered_features

def local_minima_sensitive(
    f, x, epsilon, root_dir, timeout, cex_path, use_subprocess, verifier_name,
    python_path, verifier_path, feature_ordering, perturbation=1e-6):
    """
    Algorithm 1 with features ordered by impact (least to most impactful).
    This tries to remove least impactful features first.
    """
    print("=== Algorithm 1 with Impact-Based Ordering ===")
    start_time = time.time()

    # Step 0: Order features by impact
    if feature_ordering == "naive":
        ordered_features = list(range(f.input_size))  # Original order
    elif feature_ordering == "sensitivity":
        ordered_features = order_features_by_impact(f, x, perturbation)
    else:
        err_msg = f"Unknown feature ordering method: {feature_ordering}"
        logging.error(err_msg)
        raise ValueError(err_msg)

    # Step 1: S ← [n]
    S = set(range(f.input_size))  # Start with all features
    
    full_feature_times = {}
    per_feature_time = {}
    sum_times = 0.0
    unknown_indices = set([])
    is_local_minima = True
    # Step 2: for each feature i in ordered list (least to most impactful)
    for i in ordered_features:
        # if timeout was reached, break
        if time.time() - start_time >= timeout:
            print(f"Timeout reached after {timeout} seconds. Stopping search.")
            is_local_minima = False
            break
        start_time_i = time.time()
        # Step 3: if suff(f, x, S \ {i}, ϵp) then
        S_without_i = S - {i}
        
        left_time = timeout - (time.time() - start_time)
        try:
            suff = is_sufficient_explanation(
                f, x, S_without_i, epsilon, root_dir, left_time, cex_path, 
                use_subprocess, verifier_name, python_path, verifier_path
            ) 
        except ValueError as e:
            logging.error(f"Error occurred while checking sufficiency: {e}")
            suff = False
            is_local_minima = False
        if suff:
            # Step 4: S ← S \ {i}
            S = S_without_i
            print(f"Removed feature {i} (impact-ordered). Current S: {S}")
        else:
            print(f"Cannot remove feature {i} (impact-ordered). Keeping in explanation.")
            unknown_indices.add(i)
            continue  # keep feature i in S and continue
        t_step = time.time() - start_time_i
        per_feature_time[i] = t_step
        sum_times += t_step
        full_feature_times[i] = sum_times
        # Step 5: end if (no else clause - just continue)
    # Step 6: end for
    
    # Step 7: return S
    return S, ordered_features, per_feature_time, full_feature_times, \
        unknown_indices, is_local_minima


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
    # network_path = args.network_path
    network_path = f"models/{dataset}/nam_full.pth"
    epsilon = args.epsilon
    batch_size = args.batch_size
    exp_log_dir = args.exp_log_dir
    is_bigger_nam = args.is_bigger_nam
    feature_ordering = args.feature_ordering

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

    result = {
        'dataset': dataset,
        'instance_id': int(sample_index),
        'epsilon': float(epsilon),
        'timeout': int(timeout),
        'device': device,
        'feat_sort_method': feature_ordering,
        'finished': False,
        'time': {
            # 'startup': float(startup_time), --> sorting_time
            'per_feature': []  # searching time
        }
    }

    # get the explanation
    try:
        start_time = time.time()

        # OPTION 1: ALGORITHM 1 (Original) - tries features in original order 0,1,2,3...
        # explanation = minimal_explanation_search_algorithm1(
        #    loaded_model, x, epsilon, root_dir, timeout, cex_path, use_subprocess, 
        #    verifier_name, python_path, verifier_path
        #)
        
        # OPTION 2: ALGORITHM 1 (Impact-Ordered) - tries features ordered by impact (least to most)
        explanation, ordered_features, per_feature_time, full_feature_times, \
            unknown_indices, is_local_minima = \
            local_minima_sensitive(
            loaded_model, x, epsilon, root_dir, timeout, cex_path, use_subprocess, 
            verifier_name, python_path, verifier_path, feature_ordering, perturbation=1e-6
        )

        """explanation, part1_time, part2_time, unknown_indices = global_minimal_explanation_binary_abductive(
            loaded_model, x, epsilon, root_dir, timeout, cex_path, use_subprocess, 
            verifier_name, python_path, verifier_path, compute_bounds_method
        )"""
        end_time = time.time()

        is_part = [i in explanation for i in range(loaded_model.input_size)]
        result.update({
            'feat_order': ordered_features,
            'is_part_explanation': is_part,
            'unknown_indices': list(unknown_indices)
        })
        result['time'].update({
            'per_feature': per_feature_time,
            'full_feature_times': full_feature_times
        })
        result['finished'] = is_local_minima

        # ALGORITHM 1 LOGGING: Simple time measurement
        summary = ", ".join([
                f"dataset={dataset}",
                f"sample_index={sample_index}",
                f"epsilon={epsilon}",
                f"length/total={len(explanation)}/{input_size}",
                f"Time taken: {end_time - start_time:.2f} seconds",
                f"algorithm1 minimal explanation={list(explanation)}",
                f"ordered_features={ordered_features}",
                f"per_feature_time={per_feature_time}",
                f"full_feature_times={full_feature_times}",
                f"unknown_indices={unknown_indices}",
                f"is_local_minima={is_local_minima}"
            ])
        logging.info(summary)
        print(summary)

        # save result
        import json
        result_path = f"{exp_log_dir}/result__{dataset}__{sample_index}__{epsilon}.json"
        with open(result_path, "w") as fw:
            json.dump(result, fw, indent=4)
        print(f"Result saved to {result_path}")
        logging.info(f"Result saved to {result_path}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.error(traceback.format_exc())
        print(traceback.format_exc())
    