"""Test one optimization step of the complete predictor."""

import torch

from src.models.losses import modified_mtp_loss
from src.models.predictor import MapAdaptiveTrajectoryPredictor


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two agents: target and one neighbor.
    agent_histories = torch.randn(2, 20, 2, device=device)

    # Three CCLs: two for the target and one for the neighbor.
    centerlines = torch.randn(3, 30, 2, device=device)

    # Node order:
    # 0 target, 1 neighbor, 2-3 target CCLs,
    # 4 neighbor CCL, 5 virtual target.

    stage1_edges = torch.tensor(
        [
            [4, 0, 1, 2, 3, 4, 5],
            [1, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    stage2_edges = torch.tensor(
        [
            [1, 0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    stage3_edges = torch.tensor(
        [
            [2, 3, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    prediction_edges = torch.tensor(
        [
            [0, 2, 0, 3, 0, 2, 3, 5],
            [2, 0, 3, 0, 0, 2, 3, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    mode_indices = torch.tensor(
        [2, 3, 0, 5],
        dtype=torch.long,
        device=device,
    )

    ground_truth = torch.zeros(30, 2, device=device)

    model = MapAdaptiveTrajectoryPredictor(
        input_dim=2,
        feature_dim=64,
        hidden_dim=128,
        future_steps=30,
        heads=4,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    # Save one parameter before the update.
    parameter_before = (
        model.agent_encoder.gru.weight_ih_l0.detach().clone()
    )

    optimizer.zero_grad()

    predictions = model(
        agent_histories,
        centerlines,
        stage1_edges,
        stage2_edges,
        stage3_edges,
        prediction_edges,
        mode_indices,
    )

    loss, best_mode = modified_mtp_loss(
        predictions,
        ground_truth,
    )

    loss.backward()
    optimizer.step()

    parameter_after = (
        model.agent_encoder.gru.weight_ih_l0.detach().clone()
    )

    parameter_change = (
        parameter_after - parameter_before
    ).abs().sum()

    assert predictions.shape == (4, 30, 2)
    assert parameter_change > 0

    print(f"Device: {device}")
    print(f"Predictions: {predictions.shape}")
    print(f"Best mode: {best_mode.item()}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Parameter change: {parameter_change.item():.6f}")
    print("Training-step test passed.")


if __name__ == "__main__":
    main()