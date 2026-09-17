import torch
import auto_LiRPA as lirpa
from helper.nam_train_test_save_load import FeatureNN


# Load the partial model (FeatureNN for feature i)
def load_partial_model(partial_model_path, input_dim):
    partial_model = FeatureNN(10, ())
    
    # Load the trained state_dict
    partial_model.load_state_dict(torch.load(partial_model_path, map_location="cpu"))
    partial_model.eval()
    
    return partial_model

# Define the input bounds
def compute_bounds(partial_model, x_L, x_U, be_sound, method="IBP"):  # method="CROWN" for tighter bounds

    assert torch.all(x_U >= x_L), "Upper bound must be greater than or equal to the lower bound."

    if not be_sound:
        # quick exit using sampling
        xs = torch.linspace(x_L.item(), x_U.item(), 1001)  # 1001 ensures that original value is contained
        with torch.no_grad():
            ys = partial_model(xs)
        lb = torch.min(ys)
        ub = torch.max(ys)
        return lb, ub

    # Convert the model into a bounded model
    lirpa_model = lirpa.BoundedModule(partial_model, torch.zeros_like(x_L))
    lirpa_model.eval()

    # Define a perturbation
    center = (x_U + x_L) / 2
    assert x_U - x_L >= 0
    perturbation = (x_U - x_L) / 2

    # Define a LiRPA perturbation (L-infinity norm)
    ptb = lirpa.PerturbationLpNorm(eps=perturbation, norm=float("inf"))

    # Create a bounded input tensor
    bounded_input = lirpa.BoundedTensor(center, ptb)

    # Compute bounds
    with torch.no_grad():
        lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)  # Use "CROWN" for tighter bounds
        # lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method="CROWN")  # Use "IBP" for faster bounds
    return lb, ub


if __name__ == "__main__":
    # Example usage
    partial_model_path = "models/pth/nam_partials/feature_0.pth"  # Change for different features
    input_dim = 1  # Each feature is a 1D input

    # Define example input bounds (adjust accordingly)
    x_L = torch.tensor([-1.0])  # Lower bound of input
    x_U = torch.tensor([1.0])   # Upper bound of input

    # Load model and compute bounds
    partial_model = load_partial_model(partial_model_path, input_dim)
    method = "IBP"  # Use "CROWN" for tighter bounds
    lower_bound, upper_bound = compute_bounds(partial_model, x_L, x_U, method)

    print(f"Lower Bound: {lower_bound.item()}, Upper Bound: {upper_bound.item()}")
    print("Bounds computed successfully.")
