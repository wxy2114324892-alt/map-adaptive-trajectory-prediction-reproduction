"""Trajectory decoder for multimodal motion prediction."""

import torch
from torch import nn


class TrajectoryDecoder(nn.Module):
    """Decode mode features into future XY coordinates."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        future_steps: int = 30,
    ) -> None:
        super().__init__()

        self.future_steps = future_steps

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, mode_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mode_features: Features with shape [K, feature_dim].

        Returns:
            Predicted trajectories with shape [K, future_steps, 2].
        """
        coordinates = self.mlp(mode_features)

        return coordinates.reshape(-1, self.future_steps, 2)