"""End-to-end test with simulated scene data."""

import torch

from src.models.encoders import AgentHistoryEncoder, CenterlineEncoder
from src.models.hgo import HierarchicalGraphOperator
from src.models.prediction_head import MapAdaptivePredictionHead


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulated input sequences.
    target_history = torch.randn(1, 20, 2, device=device)
    neighbor_history = torch.randn(1, 20, 2, device=device)
    target_centerlines = torch.randn(2, 30, 2, device=device)
    neighbor_centerline = torch.randn(1, 30, 2, device=device)

    # Sequence encoders.
    agent_encoder = AgentHistoryEncoder(hidden_dim=64).to(device)
    centerline_encoder = CenterlineEncoder(hidden_dim=64).to(device)

    target_feature = agent_encoder(target_history)
    neighbor_feature = agent_encoder(neighbor_history)
    target_ccl_features = centerline_encoder(target_centerlines)
    neighbor_ccl_feature = centerline_encoder(neighbor_centerline)

    # Node order: target, neighbor, target CCLs, neighbor CCL, virtual target.
    node_features = torch.cat(
        [
            target_feature,
            neighbor_feature,
            target_ccl_features,
            neighbor_ccl_feature,
            target_feature.clone(),
        ],
        dim=0,
    )

    assert node_features.shape == (6, 64)

    # Stage 1: neighbor CCL -> neighbor agent.
    stage1_edges = torch.tensor(
        [
            [4, 0, 1, 2, 3, 4, 5],
            [1, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    # Stage 2: neighbor agent -> target agent.
    stage2_edges = torch.tensor(
        [
            [1, 0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    # Stage 3: target CCLs -> target agent.
    stage3_edges = torch.tensor(
        [
            [2, 3, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    hgo = HierarchicalGraphOperator(
        feature_dim=64,
        heads=4,
    ).to(device)

    encoded_nodes = hgo(
        node_features,
        stage1_edges,
        stage2_edges,
        stage3_edges,
    )

    # Prediction edges: target agent <-> target CCLs.
    prediction_edges = torch.tensor(
        [
            [0, 2, 0, 3, 0, 2, 3, 5],
            [2, 0, 3, 0, 0, 2, 3, 5],
        ],
        dtype=torch.long,
        device=device,
    )

    # CCL modes, scene reasoning, and motion maintaining.
    mode_indices = torch.tensor(
        [2, 3, 0, 5],
        dtype=torch.long,
        device=device,
    )

    prediction_head = MapAdaptivePredictionHead(
        feature_dim=64,
        hidden_dim=128,
        future_steps=30,
        heads=4,
    ).to(device)

    trajectories = prediction_head(
        encoded_nodes,
        prediction_edges,
        mode_indices,
    )

    assert trajectories.shape == (4, 30, 2)

    print(f"Device: {device}")
    print(f"Graph nodes: {node_features.shape}")
    print(f"Encoded nodes: {encoded_nodes.shape}")
    print(f"Prediction modes: {mode_indices.tolist()}")
    print(f"Trajectories: {trajectories.shape}")
    print("Full pipeline test passed.")


if __name__ == "__main__":
    main()