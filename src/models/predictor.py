"""Complete map-adaptive trajectory predictor."""

import torch
from torch import nn

from src.models.encoders import AgentHistoryEncoder, CenterlineEncoder
from src.models.hgo import HierarchicalGraphOperator
from src.models.prediction_head import MapAdaptivePredictionHead


class MapAdaptiveTrajectoryPredictor(nn.Module):
    """Combine sequence encoders, HGO, and prediction head."""

    def __init__(
        self,
        input_dim: int = 2,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        future_steps: int = 30,
        heads: int = 4,
    ) -> None:
        super().__init__()

        self.agent_encoder = AgentHistoryEncoder(
            input_dim=input_dim,
            hidden_dim=feature_dim,
        )

        self.centerline_encoder = CenterlineEncoder(
            input_dim=input_dim,
            hidden_dim=feature_dim,
        )

        self.hgo = HierarchicalGraphOperator(
            feature_dim=feature_dim,
            heads=heads,
        )

        self.prediction_head = MapAdaptivePredictionHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            future_steps=future_steps,
            heads=heads,
        )

    def forward(
        self,
        agent_histories: torch.Tensor,
        centerlines: torch.Tensor,
        stage1_edges: torch.Tensor,
        stage2_edges: torch.Tensor,
        stage3_edges: torch.Tensor,
        prediction_edges: torch.Tensor,
        mode_indices: torch.Tensor,
        target_agent_index: int = 0,
    ) -> torch.Tensor:
        """Predict multimodal trajectories for one scene."""

        agent_features = self.agent_encoder(agent_histories)
        centerline_features = self.centerline_encoder(centerlines)

        target_feature = agent_features[
            target_agent_index : target_agent_index + 1
        ]

        # Agent nodes, CCL nodes, and the virtual target node.
        node_features = torch.cat(
            [
                agent_features,
                centerline_features,
                target_feature.clone(),
            ],
            dim=0,
        )

        encoded_nodes = self.hgo(
            node_features,
            stage1_edges,
            stage2_edges,
            stage3_edges,
        )

        return self.prediction_head(
            encoded_nodes,
            prediction_edges,
            mode_indices,
        )