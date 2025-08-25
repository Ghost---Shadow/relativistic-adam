"""
Demo: Exploding Gradients - Adam vs RelativisticAdam
This demo shows how a high learning rate causes gradient explosion with regular Adam,
while RelativisticAdam handles it gracefully due to its relativistic speed limit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from relativistic_adam import RelativisticAdam

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class DeepNet(nn.Module):
    """
    A reasonably deep network that can exhibit gradient explosion with high learning rates.
    """

    def __init__(self, input_dim=10, hidden_dim=128, output_dim=1, num_layers=8):
        super(DeepNet, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            # No batch norm - makes it easier to explode

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize with slightly larger than normal weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.network(x)


def create_dataset(n_samples=100, n_features=10, noise_level=0.1):
    """
    Create a synthetic regression dataset.
    """
    # Generate random input features
    X = torch.randn(n_samples, n_features)

    # Create a non-linear target with some noise
    # True function: sum of squares plus interactions
    y = torch.sum(X**2, dim=1, keepdim=True)
    y = y + 0.5 * torch.sum(X[:, :5] * X[:, 5:], dim=1, keepdim=True)
    y = y + noise_level * torch.randn(n_samples, 1)

    # Normalize targets to reasonable range
    y = (y - y.mean()) / (y.std() + 1e-8)

    return X, y


def train_model(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=200,
    model_name="Model",
    print_freq=20,
):
    """
    Train the model and track metrics.
    """
    train_losses = []
    val_losses = []
    grad_norms = []
    param_norms = []

    exploded = False
    explosion_epoch = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(X_train)
        train_loss = F.mse_loss(output, y_train)

        # Backward pass
        train_loss.backward()

        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        # Calculate parameter norm
        param_norm = 0.0
        for p in model.parameters():
            param_norm += p.data.norm(2).item() ** 2
        param_norm = param_norm**0.5

        # Check for explosion
        if (
            not np.isfinite(train_loss.item())
            or not np.isfinite(total_norm)
            or total_norm > 1e10
        ):
            if not exploded:
                explosion_epoch = epoch
                exploded = True
                print(f"\n💥 {model_name} EXPLODED at epoch {epoch}!")
                print(
                    f"   Loss: {train_loss.item():.2e}, Gradient norm: {total_norm:.2e}"
                )

            # Record as NaN/Inf
            train_losses.append(
                float("inf") if np.isfinite(train_loss.item()) else float("nan")
            )
            val_losses.append(float("inf"))
            grad_norms.append(float("inf") if np.isfinite(total_norm) else float("nan"))
            param_norms.append(
                float("inf") if np.isfinite(param_norm) else float("nan")
            )

            # Skip the optimizer step
            continue
        else:
            # Record metrics
            train_losses.append(train_loss.item())
            grad_norms.append(total_norm)
            param_norms.append(param_norm)

        # Optimizer step (only if not exploded)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = F.mse_loss(val_output, y_val)
            val_losses.append(
                val_loss.item() if np.isfinite(val_loss.item()) else float("inf")
            )

        # Print progress
        if epoch % print_freq == 0 or (exploded and explosion_epoch == epoch):
            if not exploded:
                print(
                    f"{model_name} - Epoch {epoch:3d}: "
                    f"Train Loss = {train_loss.item():.4f}, "
                    f"Val Loss = {val_loss.item():.4f}, "
                    f"Grad Norm = {total_norm:.2e}"
                )

    if not exploded:
        print(f"\n✅ {model_name} completed training successfully!")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "exploded": exploded,
        "explosion_epoch": explosion_epoch,
    }


def main():
    print("=" * 80)
    print("Gradient Explosion Demo: High Learning Rate")
    print("Adam vs RelativisticAdam")
    print("=" * 80)
    print("\nThis demo uses a high learning rate to cause gradient explosion.\n")

    # Create dataset
    X, y = create_dataset(n_samples=200, n_features=10)

    # Split into train/validation
    n_train = 150
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    print(
        f"Dataset created: {n_train} training samples, {len(X_val)} validation samples"
    )
    print(f"Input dimension: {X.shape[1]}, Output dimension: 1\n")

    # Create two identical models
    print("Creating two identical deep networks...")
    model_adam = DeepNet(input_dim=10, hidden_dim=128, output_dim=1, num_layers=8)
    model_relativistic = DeepNet(
        input_dim=10, hidden_dim=128, output_dim=1, num_layers=8
    )

    # Copy weights to ensure identical initialization
    model_relativistic.load_state_dict(model_adam.state_dict())

    total_params = sum(p.numel() for p in model_adam.parameters())
    print(f"Network architecture: 8 layers, {total_params:,} parameters")
    print("Networks initialized with identical weights\n")

    # High learning rate that will cause explosion
    lr = 1.0  # Very high learning rate!
    print(f"⚠️  Using VERY HIGH learning rate: {lr}")
    print("   (Normal would be 0.001 - 0.01)\n")

    # Create optimizers
    optimizer_adam = Adam(model_adam.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    optimizer_relativistic = RelativisticAdam(
        model_relativistic.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        speed_limit=0.1,  # This will prevent explosion
        relativistic_mode="per_param",
    )

    print("=" * 80)
    print("Training with Standard Adam (high LR = explosion expected)...")
    print("=" * 80)
    results_adam = train_model(
        model_adam,
        optimizer_adam,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=200,
        model_name="Adam",
        print_freq=20,
    )

    print("\n" + "=" * 80)
    print("Training with RelativisticAdam (protected by speed limit)...")
    print("=" * 80)
    results_relativistic = train_model(
        model_relativistic,
        optimizer_relativistic,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=200,
        model_name="RelativisticAdam",
        print_freq=20,
    )

    # Plotting
    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Gradient Explosion with High Learning Rate (LR={lr})",
        fontsize=14,
        fontweight="bold",
    )

    # Colors
    adam_color = "#E74C3C"  # Red
    rel_color = "#3498DB"  # Blue

    # Plot 1: Training Loss
    axes[0, 0].semilogy(
        results_adam["train_losses"],
        color=adam_color,
        label="Adam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 0].semilogy(
        results_relativistic["train_losses"],
        color=rel_color,
        label="RelativisticAdam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training Loss (log)")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].semilogy(
        results_adam["val_losses"],
        color=adam_color,
        label="Adam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 1].semilogy(
        results_relativistic["val_losses"],
        color=rel_color,
        label="RelativisticAdam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation Loss (log)")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Gradient Norms
    axes[0, 2].semilogy(
        results_adam["grad_norms"],
        color=adam_color,
        label="Adam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 2].semilogy(
        results_relativistic["grad_norms"],
        color=rel_color,
        label="RelativisticAdam",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Gradient Norm (log)")
    axes[0, 2].set_title("Gradient Norms")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Parameter Norms
    axes[1, 0].plot(
        results_adam["param_norms"],
        color=adam_color,
        label="Adam",
        linewidth=2,
        alpha=0.8,
    )
    axes[1, 0].plot(
        results_relativistic["param_norms"],
        color=rel_color,
        label="RelativisticAdam",
        linewidth=2,
        alpha=0.8,
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Parameter Norm")
    axes[1, 0].set_title("Parameter Norms")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Gradient norm ratio (first 50 epochs)
    max_epochs = min(
        50, len(results_adam["grad_norms"]), len(results_relativistic["grad_norms"])
    )
    if max_epochs > 0:
        adam_grads = np.array(results_adam["grad_norms"][:max_epochs])
        rel_grads = np.array(results_relativistic["grad_norms"][:max_epochs])

        # Replace inf/nan with large number for ratio calculation
        adam_grads_clean = np.where(np.isfinite(adam_grads), adam_grads, 1e10)
        rel_grads_clean = np.where(np.isfinite(rel_grads), rel_grads, 1.0)

        ratio = adam_grads_clean / (rel_grads_clean + 1e-8)

        axes[1, 1].semilogy(ratio, color="purple", linewidth=2, alpha=0.8)
        axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Gradient Ratio (Adam/Relativistic)")
        axes[1, 1].set_title("Gradient Norm Ratio (First 50 Epochs)")
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary
    axes[1, 2].axis("off")

    # Calculate final metrics
    adam_final_loss = (
        results_adam["train_losses"][-1]
        if results_adam["train_losses"]
        else float("nan")
    )
    rel_final_loss = (
        results_relativistic["train_losses"][-1]
        if results_relativistic["train_losses"]
        else float("nan")
    )

    adam_max_grad = max(
        [g for g in results_adam["grad_norms"] if np.isfinite(g)], default=0
    )
    rel_max_grad = max(
        [g for g in results_relativistic["grad_norms"] if np.isfinite(g)], default=0
    )

    summary_text = f"""
    RESULTS SUMMARY
    {'-' * 40}
    
    Learning Rate: {lr} (extremely high!)
    
    Standard Adam:
    • Exploded: {results_adam['exploded']}
    • Explosion Epoch: {results_adam['explosion_epoch'] if results_adam['exploded'] else 'N/A'}
    • Max Gradient Norm: {adam_max_grad:.2e}
    • Final Loss: {'EXPLODED' if not np.isfinite(adam_final_loss) else f'{adam_final_loss:.4f}'}
    
    RelativisticAdam:
    • Exploded: {results_relativistic['exploded']}
    • Speed Limit: 0.1
    • Max Gradient Norm: {rel_max_grad:.2e}
    • Final Loss: {'EXPLODED' if not np.isfinite(rel_final_loss) else f'{rel_final_loss:.4f}'}
    
    Outcome:
    • Gradient reduction: {adam_max_grad/rel_max_grad:.1f}x
    • Winner: {'RelativisticAdam (prevented explosion!)' if results_adam['exploded'] and not results_relativistic['exploded'] else 'Both stable' if not results_adam['exploded'] else 'Both exploded'}
    """

    axes[1, 2].text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save figure
    output_file = "high_lr_explosion_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if results_adam["exploded"] and not results_relativistic["exploded"]:
        print("✅ SUCCESS! RelativisticAdam prevented gradient explosion!")
        print(f"   Adam exploded at epoch {results_adam['explosion_epoch']}")
        print(f"   RelativisticAdam remained stable throughout training")
        print(
            f"   The relativistic speed limit successfully protected against high LR!"
        )
    elif not results_adam["exploded"]:
        print(
            "⚠️  Adam didn't explode. Try an even higher learning rate (e.g., lr=10.0)"
        )
    else:
        print(
            "Both optimizers had issues. Consider adjusting the speed_limit parameter."
        )

    print(f"\nWith LR={lr}, standard Adam cannot handle the large gradient updates,")
    print("while RelativisticAdam's speed limit acts as automatic gradient clipping.")
    print("\nCheck 'high_lr_explosion_comparison.png' for detailed visualizations.")


if __name__ == "__main__":
    main()
