import torch

def convert_and_save_model_to_onnx(model, path, input_size=(1, 1, 28, 28)):
    model.eval()
    dummy_input = torch.randn(*input_size)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13
    )