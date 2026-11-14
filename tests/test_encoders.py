"""Tests for the GRU sequence encoders."""

import torch

from src.models.encoders import AgentHistoryEncoder, CenterlineEncoder


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulated agent histories: [agents, time steps, coordinates]
    agent_history = torch.randn(8, 20, 2, device=device)

    # Simulated centerlines: [centerlines, points, coordinates]
    centerlines = torch.randn(12, 30, 2, device=device)

    agent_encoder = AgentHistoryEncoder(hidden_dim=64).to(device)
    centerline_encoder = CenterlineEncoder(hidden_dim=64).to(device)

    agent_features = agent_encoder(agent_history)
    centerline_features = centerline_encoder(centerlines)

    assert agent_features.shape == (8, 64)
    assert centerline_features.shape == (12, 64)

    print(f"Device: {device}")
    print(f"Agent features: {agent_features.shape}")
    print(f"Centerline features: {centerline_features.shape}")
    print("All encoder tests passed.")


if __name__ == "__main__":
    main()