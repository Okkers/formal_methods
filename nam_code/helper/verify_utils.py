def verify(sub_model_path, vnnlib_path, root_dir, timeout=60, cex_path="cex.txt",
           use_subprocess=True, verifier_name='abcrown',
           python_path="", verifier_path="", input_shape="[1,]"):
    if verifier_name == 'abcrown':
        from dnnv_tools.abcrown_utils import generate_abcrown_yaml_file, solve_with_abcrown
        yaml_path = f"{root_dir}/abcrown.yaml"
        # some call sites pass 'root_dir' to YAML gen, some don't.
        try:
            generate_abcrown_yaml_file(
                yaml_path, sub_model_path, vnnlib_path, timeout,
                input_shape=input_shape, root_dir=root_dir
            )
        except TypeError:
            generate_abcrown_yaml_file(
                yaml_path, sub_model_path, vnnlib_path, timeout,
                input_shape=input_shape
            )
        res, cex = solve_with_abcrown(
            yaml_path, cex_path, timeout, use_subprocess, python_path, verifier_path
        )
    elif verifier_name == 'marabou':
        from dnnv_tools.marabou_utils import solve_with_marabou
        res, cex, _ = solve_with_marabou(sub_model_path, vnnlib_path)
    else:
        raise ValueError(f"Unknown verifier name: {verifier_name}")
    return res, cex