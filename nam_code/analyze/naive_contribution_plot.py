import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.curdir)))
import math
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_feature_effects(N, x, lower, upper, num_points=10, epsilon=0.1, device="cpu"):
    """
    Plot feature effect curves for a Neural Additive Model (NAM).

    Parameters:
        N        : NeuralAdditiveModel (callable, takes tensor [1, d])
        x        : input tensor of shape (d,)
        lower    : array-like of shape (d,), lower bounds
        upper    : array-like of shape (d,), upper bounds
        num_points  : number of points between lower and upper for feature perturbations
        device   : "cpu" or "cuda"
    """
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = 0 * x;
    d = x.shape[0]

    # fig, axes = plt.subplots(
    #     nrows=(d+2)//4, ncols=4, figsize=(18, 4*((d+2)//4))
    # )
    cols = 6
    rows = math.ceil(d / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), constrained_layout=True)
    # fig, axes = plt.subplots(nrows=(d+2)//3, ncols=3, figsize=(15, 5*((d+2)//3)))
    axes = axes.flatten()
    jump = (upper - lower) / num_points
    span = upper-lower
    # epsilon = np.min(jump[jump > 0]) if np.any(jump > 0) else 0.1  # avoid zero jump

    for i in range(d):
        subnet = N.feature_nns[i]

        with torch.no_grad():
            # generate exact value for the i-th feature of x
            xi = x[i].unsqueeze(0)
            yi = (subnet(xi) * N.feature_weights[i])
            y[i] = yi

            # generate entire range of values for the i-th feature
            xi_range = torch.arange(lower[i], upper[i], jump[i])
            yi_range = subnet(xi_range) * N.feature_weights[i]

            # generate epsilon range of values for the i-th feature
            xi_epsilon = torch.arange(xi.item()-span[i]*epsilon,xi.item()+span[i]*epsilon, jump[i])
            yi_epsilon = subnet(xi_epsilon) * N.feature_weights[i]
            lower[i] = torch.min(yi_epsilon).item()
            upper[i] = torch.max(yi_epsilon).item()

        print("---")
        print(f"Feature {i}: x={xi.item()}, bounds=[{lower[i]},{upper[i]}], epsilon region=[{xi_epsilon[0]},{xi_epsilon[-1]}].\n Values: ")
        print(xi_range)
        print(yi_range.T)

        axes[i].plot(xi_range.detach().numpy(), yi_range.detach().numpy(), marker="o")
        axes[i].plot(xi_epsilon.detach().numpy(), yi_epsilon.detach().numpy(), marker="o")
        axes[i].plot(x[i], yi.detach().numpy(), marker="o")
        axes[i].set_title(f"x_{i}", fontsize=12)
        axes[i].set_xlabel(f"x[{i}]", fontsize=12)
        axes[i].set_ylabel("N(x)")
        axes[i].tick_params(axis="x", rotation=30)

    # hide unused subplots
    for j in range(d, len(axes)):
        axes[j].axis("off")

    filepath = f"{results_dir}/feature_effects_sample_{sample_index}_num_steps_{num_points}.png"
    print(f"Saving feature effects plot to {filepath}")
    fig.savefig(filepath)
    plt.close(fig)

    fig = plt.figure()

    # Compute asymmetric error
    yerr = [[v - l for v, l in zip(y, lower)],  # lower errors
        [u - v for v, u in zip(y, upper)]]  # upper errors

    features = [f"x_{i}" for i in range(0, d)]
    plt.errorbar(features, y, yerr=yerr, fmt='o', capsize=5, color="tab:blue")
    plt.ylabel("Value")
    plt.title("Features with Bounds")

    filepath = f"{results_dir}/feature_effects_sample_{sample_index}_num_steps_{num_points}_epsilon.png"
    print(f"Saving feature effects plot to {filepath}")
    fig.savefig(filepath)
    plt.close(fig)

    # plt.show()
    return fig


def plot_feature_contribution_ranges(N, x, lower, upper, num_points=10, device="cpu"):
    """
    Compute output ranges when varying one feature at a time.
    
    Args:
        N: model (NeuralAdditiveModel)
        x: input tensor
        lower: tensor of lower bounds (same shape as x)
        upper: tensor of upper bounds
        steps: number of steps between [lower_i, upper_i]
        device: cpu/cuda
    Returns:
        values_per_feature: list of output arrays, one per feature
    """
    x = x.to(device)
    lower, upper = torch.Tensor(lower).to(device), torch.Tensor(upper).to(device)
    d = x.shape[0]
    values_per_feature = []

    for i in range(d):
        xi_values = torch.linspace(lower[i], upper[i], num_points).to(device)
        outputs = []
        for val in xi_values:
            x_mod = x.clone()
            x_mod[i] = val
            y = N(x_mod.unsqueeze(0))  # batch of size 1
            outputs.append(y.item())
        values_per_feature.append(outputs)
    # Plot per-feature boxplots of contribution ranges.
    
    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(values_per_feature, positions=np.arange(len(values_per_feature)))
    plt.xlabel("Feature index")
    plt.ylabel("Model output")
    plt.title("Feature contribution ranges")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Script for managing YAML generation and verification")
    # Arguments
    parser.add_argument('--root_dir', type=str, default="./abcrown_dir",
                        help="Root directory for the project.")
    parser.add_argument('--dataset', type=str, default="heloc",  # breast_cancer
                        help="Name of the dataset.")
    parser.add_argument('--sample_index', type=int, default=0,
                        help="Index of sample in dataset (default: 0).")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Size of each batch from the dataset (default: 1).")
    parser.add_argument('--network_path', type=str, default="models/pth/nam_full.pth",
                        help="Path to the neural network model.")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help="Perturbation bound for verification.")
    parser.add_argument('--num_points', type=int, default=100,
                        help="Number of points for feature perturbations.")
    parser.add_argument('--device', type=str, default="cpu",
                        help="device type to use (cpu/gpu).")
    parser.add_argument('--contribution_plots_dir', type=str, default="contribution_plots_dir",
                        help="Relative path to the directory for contribution plots.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    device = args.device
    sample_index = args.sample_index
    dataset = args.dataset
    batch_size = args.batch_size
    # network_path = args.network_path
    network_path = f"../models/big/{dataset}/nam_full.pth"
    epsilon = args.epsilon
    num_points = args.num_points

    contribution_plots_dir = args.contribution_plots_dir
    results_dir = f"{contribution_plots_dir}/{dataset}"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # load the full model
    from helper.nam_train_test_save_load import load_full_model, load_data
    train_loader, test_loader, input_size = load_data(dataset, batch_size=batch_size)
    loaded_model = load_full_model(network_path, input_size, device, True)
    # test_model(loaded_model, test_loader, device)

    # load the input
    X_batch, y_batch = list(test_loader)[0]  # change to [sample_index] if batch_size == 1
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    # x = X_batch[0]
    x = torch.tensor([0.66,0.2,0.1,0.4,0.35,0.3,0.4,0.025,0.4,0.3,0.5,0.5,0.2,0.25,0.3,0.1,0.05,0.3,0.3,0.3,0.3])

    bounds = [x-epsilon, x+epsilon]
    lower = torch.min(X_batch, dim=0).values
    upper = torch.max(X_batch, dim=0).values
    
    # plot per feature: effect curves
    plot_feature_effects(loaded_model, x, lower, upper, num_points, epsilon, device="cpu")


    # plot per feature: contribution ranges
    # fig = plot_feature_contribution_ranges(loaded_model, x, lower, upper, num_points, device="cpu")
    # filepath = f"{results_dir}/feature_contribution_ranges_sample_{sample_index}_num_steps_{num_points}.png"
    # print(f"Saving feature contribution ranges plot to {filepath}")
    # fig.savefig(filepath)
    # plt.close(fig)

