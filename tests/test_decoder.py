"""Test for the trajectory decoder."""

import torch

from src.models.decoder import TrajectoryDecoder


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Four candidate modes, each represented by a 64-D feature.
    mode_features = torch.randn(4, 64, device=device)

    decoder = TrajectoryDecoder(
        feature_dim=64,
        hidden_dim=128,
        future_steps=30,
    ).to(device)

    trajectories = decoder(mode_features)

    assert trajectories.shape == (4, 30, 2)

    print(f"Device: {device}")
    print(f"Mode features: {mode_features.shape}")
    print(f"Predicted trajectories: {trajectories.shape}")
    print("Trajectory decoder test passed.")


if __name__ == "__main__":
    main()