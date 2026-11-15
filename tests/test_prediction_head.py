"""Test for the map-adaptive prediction head."""

import torch

from src.models.prediction_head import MapAdaptivePredictionHead


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulated HGO output: six nodes with 64-D features.
    node_features = torch.randn(6, 64, device=device)

    # The virtual target starts from the target dynamics feature.
    node_features[5] = node_features[0].clone()

    # Target agent <-> target CCLs, plus required self-loops.
    prediction_edges = torch.tensor(
        [
            [0, 2, 0, 3, 0, 2, 3, 5],
            [2, 0, 3, 0, 0, 2, 3, 5],
        ],
        device=device,
    )

    # CCL 1, CCL 2, scene reasoning, motion maintaining.
    mode_indices = torch.tensor(
        [2, 3, 0, 5],
        device=device,
    )

    prediction_head = MapAdaptivePredictionHead(
        feature_dim=64,
        hidden_dim=128,
        future_steps=30,
        heads=4,
    ).to(device)

    trajectories = prediction_head(
        node_features,
        prediction_edges,
        mode_indices,
    )

    assert trajectories.shape == (4, 30, 2)

    print(f"Device: {device}")
    print(f"Selected modes: {mode_indices.tolist()}")
    print(f"Predicted trajectories: {trajectories.shape}")
    print("Prediction head test passed.")


if __name__ == "__main__":
    main()