"""Hierarchical Graph Operator."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class HierarchicalGraphOperator(nn.Module):
    """Encode a scene graph through three masked GAT stages."""

    def __init__(self, feature_dim: int = 64, heads: int = 4) -> None:
        super().__init__()

        self.stage1 = GATConv(
            feature_dim,
            feature_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )

        self.stage2 = GATConv(
            feature_dim,
            feature_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )

        self.stage3 = GATConv(
            feature_dim,
            feature_dim,
            heads=heads,
            concat=False,
            add_self_loops=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        stage1_edges: torch.Tensor,
        stage2_edges: torch.Tensor,
        stage3_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the three information-flow stages."""

        features = F.elu(self.stage1(node_features, stage1_edges))
        features = F.elu(self.stage2(features, stage2_edges))
        features = F.elu(self.stage3(features, stage3_edges))

        return features