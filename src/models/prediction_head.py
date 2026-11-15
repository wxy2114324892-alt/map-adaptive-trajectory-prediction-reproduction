"""Map-adaptive multimodal prediction head."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from src.models.decoder import TrajectoryDecoder


class MapAdaptivePredictionHead(nn.Module):
    """Generate trajectories from selected graph nodes."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        future_steps: int = 30,
        heads: int = 4,
    ) -> None:
        super().__init__()

        self.prediction_gat = GATConv(
            feature_dim,
            feature_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )

        self.decoder = TrajectoryDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            future_steps=future_steps,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        prediction_edges: torch.Tensor,
        mode_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one trajectory for each selected mode node."""

        node_features = F.elu(
            self.prediction_gat(node_features, prediction_edges)
        )

        mode_features = node_features[mode_indices]

        return self.decoder(mode_features)