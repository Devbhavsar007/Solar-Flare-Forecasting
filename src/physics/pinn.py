"""
Neupert Effect PINN (Physics-Informed Neural Network).

Encodes the physical constraint that hard X-ray (HXR) emission is
proportional to the time derivative of soft X-ray (SXR) flux during
the impulsive phase of a solar flare (Neupert Effect, 1968).

The physics loss penalises cases where dSXR/dt and HXR have opposing
signs, which would violate the Neupert relationship.
"""
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
import torch.nn as nn

try:
    import mlflow
except ImportError:
    mlflow = None


class NeupertPINN(nn.Module):
    """
    Physics-Informed Neural Network for the Neupert Effect.

    Takes a window of SXR flux values and predicts dSXR/dt.
    The physics loss enforces that dSXR/dt * HXR >= 0 during flare rise.
    """

    def __init__(self, input_dim: int = 60, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # predicts scalar dSXR/dt
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict dSXR/dt from an SXR flux window.

        Args:
            x: Tensor of shape (batch, input_dim) containing SXR flux values.

        Returns:
            Tensor of shape (batch, 1) with predicted dSXR/dt.
        """
        return self.net(x)


def violation_penalty(dSXR_dt: torch.Tensor, HXR: torch.Tensor) -> torch.Tensor:
    """
    Compute the Neupert Effect violation penalty.

    The Neupert Effect states that HXR ∝ dSXR/dt, so their product
    should be non-negative during the impulsive phase of a flare.
    A negative product indicates a physical violation.

    Args:
        dSXR_dt: Predicted time derivative of SXR flux (batch,) or (batch, 1).
        HXR: Observed HXR flux values (batch,) or (batch, 1).

    Returns:
        Scalar tensor: mean penalty (>= 0). Zero if no violations.
    """
    product = dSXR_dt * HXR
    violation = -torch.clamp(product, max=0)  # only penalise negative products
    return violation.mean()


def train_neupert_pinn(
    X_windows: torch.Tensor,
    hxr_windows: torch.Tensor,
    sxr_windows: torch.Tensor,
    beta: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-3,
) -> NeupertPINN:
    """
    Train the Neupert PINN.

    Args:
        X_windows: SXR flux windows (batch, window_len) — model input.
        hxr_windows: HXR flux values (batch,) — for physics constraint.
        sxr_windows: Actual dSXR/dt targets (batch,) — for data loss.
        beta: Weight for the physics violation penalty.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Trained NeupertPINN model.
    """
    input_dim = X_windows.shape[1]
    pinn = NeupertPINN(input_dim=input_dim)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        predicted_dSXR_dt = pinn(X_windows).squeeze(-1)

        # Data-driven loss
        physics_loss = mse_loss(predicted_dSXR_dt, sxr_windows)

        # Physics-informed penalty
        penalty = violation_penalty(predicted_dSXR_dt, hxr_windows)

        total_loss = physics_loss + beta * penalty

        total_loss.backward()
        optimizer.step()

        # Log to MLflow every epoch
        if mlflow is not None:
            try:
                mlflow.log_metrics(
                    {
                        "physics_loss": physics_loss.item(),
                        "violation_penalty": penalty.item(),
                        "total_loss": total_loss.item(),
                    },
                    step=epoch,
                )
            except Exception:
                pass  # MLflow may not be active; don't crash training

    # Save model weights
    import os
    os.makedirs("models", exist_ok=True)
    torch.save(pinn.state_dict(), "models/neupert_pinn.pt")

    return pinn
