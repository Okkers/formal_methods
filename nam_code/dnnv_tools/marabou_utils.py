from maraboupy import Marabou


def solve_with_marabou(onnx_model_path, vnnlib_path):
    network = Marabou.read_onnx(onnx_model_path)
    options = Marabou.createOptions()
    res, cs, stats = network.solve(options=options, propertyFilename=str(vnnlib_path))
    return res, cs, stats
