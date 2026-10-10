import subprocess
import sys
import torch
import os

abcrown_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "abcrown_dir"))
auto_lirpa_dir = os.path.join(abcrown_dir, "auto_LiRPA")
if auto_lirpa_dir not in sys.path:
    sys.path.insert(0, auto_lirpa_dir)
if abcrown_dir not in sys.path:
    sys.path.append(abcrown_dir)

from complete_verifier import ABCROWN
import arguments


def add_general(yaml_file, device="cpu", root_dir="./abcrown_dir", tab=" " * 2, verbosity=0):
    yaml_file.write("\n".join([
        "general:\n"
        f"{tab}device: {device}",
        f"{tab}loss_reduction_func: min",
        f"{tab}root_path: {os.path.abspath(root_dir)}",
        f"{tab}csv_name: instances.csv",
        f"{tab}graph_optimizer: Customized('custom_graph_optimizer', 'merge_sign')",
        f"{tab}sparse_interm: false",
        f"{tab}save_adv_example: true"
    ]) + "\n")


def add_model(yaml_file, tab=" " * 2, input_shape="[1,]"):
    # print("*"*40)
    # print(f"add_model(): input_shape={input_shape}")
    # print("*"*40)
    yaml_file.write("\n".join([
        "model:",
        # f"{tab}input_shape: [1, 1,28,28]",
        f"{tab}input_shape: {str(input_shape).replace('(', '[').replace(')', ']')}",
        f"{tab}onnx_loader: Customized('custom_model_loader', 'customized_my_loader')",
    ]) + "\n")


def add_attack(yaml_file, tab=" " * 2):
    yaml_file.write("\n".join([
        "attack:",
        f"{tab}pgd_order: before",
        f"{tab}pgd_restarts: 2",
        f"{tab}pgd_batch_size: 2",
        f"{tab}cex_path: ./cex.txt",
    ]) + "\n")


def add_solver(yaml_file, tab=" " * 2):
    yaml_file.write("\n".join([
        "solver:",
        f"{tab}batch_size: 1",
        f"{tab}min_batch_size_ratio: 1",
        f"{tab}alpha-crown:",
        f"{tab}{tab}disable_optimization: ['MaxPool']",
        f"{tab}beta-crown:",
        f"{tab}{tab}iteration: 20",
        f"{tab}{tab}lr_beta: 0.03",
        f"{tab}mip:",
        f"{tab}{tab}parallel_solvers: 8",
        f"{tab}{tab}solver_threads: 4",
        f"{tab}{tab}refine_neuron_time_percentage: 0.8",
        f"{tab}{tab}skip_unsafe: True",
    ]) + "\n")


def add_bab(yaml_file, timeout, tab=" " * 2):
    # yaml_file.write("\n".join([
    #     "bab:",
    #     f"{tab}initial_max_domains: 100",
    #     f"{tab}interm_transfer: False",
    #     f"{tab}branching:",
    #     f"{tab}{tab}method: kfsb",
    #     f"{tab}{tab}candidates: 7",
    #     f"{tab}{tab}reduceop: max",
    # ]) + "\n")
    yaml_file.write("\n".join([
        "bab:",
        f"{tab}timeout: {timeout}",
        f"{tab}pruning_in_iteration: False",
        f"{tab}sort_domain_interval: 1",
        f"{tab}branching:",
        f"{tab}{tab}method: nonlinear",
        f"{tab}{tab}candidates: 3",
        f"{tab}{tab}nonlinear_split:",
        f"{tab}{tab}{tab}num_branches: 2",
        f"{tab}{tab}{tab}method: shortcut",
        f"{tab}{tab}{tab}filter: true"
    ]) + "\n")


def generate_instances_file(root_dir, sub_model_path, vnnlib_path, timeout):
    with open(f"{root_dir}/instances.csv", "w") as instances_fw:
        # use only relative paths from root_dir in the instances.csv file
        sub_model_relpath = sub_model_path[len(root_dir) + 1:]  # +1 removes leading '/'
        vnnlib_relpath = vnnlib_path[len(root_dir) + 1:]  # +1 removes leading '/'
        instances_fw.write(f"{sub_model_relpath},{vnnlib_relpath},{timeout}\n")
        

def generate_abcrown_yaml_file(
    yaml_path, sub_model_path, vnnlib_path, timeout, input_shape="[1,]", root_dir="./abcrown_dir"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root_dir = "./abcrown_dir"
    generate_instances_file(root_dir, sub_model_path, vnnlib_path, timeout)
    with open(yaml_path, "w") as yaml_fw:
        add_general(yaml_fw, device, root_dir)
        add_model(yaml_fw, input_shape=input_shape)
        add_attack(yaml_fw)
        add_solver(yaml_fw)
        add_bab(yaml_fw, timeout)


def get_result(abcrown):
    v_summary = abcrown.logger.verification_summary
    unsafe_pgd = v_summary.get("unsafe-pgd") is not None or v_summary.get("unsafe-pgd (timed out)") is not None
    unsafe_bab = v_summary.get("unsafe-bab") is not None
    safe_incomplete = v_summary.get("safe-incomplete") is not None or v_summary.get("safe-incomplete (timed out)") is not None
    safe = v_summary.get("safe") is not None
    safe = safe or safe_incomplete
    unsafe = unsafe_pgd or unsafe_bab
    unknown = v_summary.get("unknown") is not None
    assert (int(safe) + int(unsafe) + int(unknown) == 1)
    return "unsafe" if unsafe else "safe" if safe else "unknown"


def extract_ce(cex_file):
    with open(cex_file) as fr:
        cex = [
            float(line.split(" ")[-1].split(")")[0])
            for line in fr if "X_" in line
        ]
    return torch.Tensor(cex)


def extract_res_ce(output, output_path):
    if output.split("\n")[-1].startswith("safe"):
        return "safe", None
    elif output.split("\n")[-1].startswith("unsafe"):
        cex = extract_ce(output_path)
        return "unsafe", cex


def solve_with_abcrown(yaml_path, output_path, timeout, use_subprocess, python_path, verifier_path):
    if use_subprocess:
        command = f"{python_path} {verifier_path} --config {yaml_path}"
        result = subprocess.run(command, capture_output=True, text=True, shell=True, timeout=timeout)
        output = result.stdout.strip()
        res, cex = extract_res_ce(output, output_path)
    else:
        try:
            from complete_verifier import abcrown
            abcrown = abcrown.ABCROWN(args=[
                "--config", yaml_path,
                # "--verbose", "0",     # minimal output
                "--timeout", "20"    # total verification timeout in seconds
            ])
            abcrown.main()
            res = get_result(abcrown)
            if res in ["safe", "unknown"]:
                cex = None
            elif res == "unsafe":
                cex_file = './cex.txt'
                cex = extract_ce(cex_file)
            elif res == "unknown":
                cex = None
            else:
                raise ValueError(f"Unknown verifier result: {res}")
        except RuntimeError as e:
            print(f"RuntimeError in ABCROWN: {e}")
            import traceback
            traceback.print_exc()
            res = "unknown"
            cex = None
    return res, cex
