"""GRU encoders for agent histories and candidate centerlines."""

import torch
from torch import nn


class GRUSequenceEncoder(nn.Module):
    """Encode a sequence into a fixed-length feature vector."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()

        # Input shape: [batch_size, sequence_length, input_dim]
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: Input tensor of shape [B, T, input_dim].

        Returns:
            Encoded feature of shape [B, hidden_dim].
        """
        _, final_hidden = self.gru(sequence)

        # Select the final hidden state from the last GRU layer.
        return final_hidden[-1]


class AgentHistoryEncoder(GRUSequenceEncoder):
    """Encode an agent's historical motion states."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )


class CenterlineEncoder(GRUSequenceEncoder):
    """Encode the ordered points of a candidate centerline."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )