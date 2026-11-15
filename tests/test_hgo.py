"""Test for the Hierarchical Graph Operator."""

import torch

from src.models.hgo import HierarchicalGraphOperator


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Six graph nodes, each represented by a 64-D feature.
    node_features = torch.randn(6, 64, device=device)
    node_features[5] = node_features[0].clone()

    # Stage 1: neighbor CCL -> neighbor agent.
    stage1_edges = torch.tensor(
        [
            [4, 0, 1, 2, 3, 4, 5],
            [1, 0, 1, 2, 3, 4, 5],
        ],
        device=device,
    )

    # Stage 2: neighbor agent -> target agent.
    stage2_edges = torch.tensor(
        [
            [1, 0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 4, 5],
        ],
        device=device,
    )

    # Stage 3: target CCLs -> target agent.
    stage3_edges = torch.tensor(
        [
            [2, 3, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 1, 2, 3, 4, 5],
        ],
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

    assert encoded_nodes.shape == (6, 64)

    print(f"Device: {device}")
    print(f"Input nodes: {node_features.shape}")
    print(f"Encoded nodes: {encoded_nodes.shape}")
    print("HGO test passed.")


if __name__ == "__main__":
    main()