"""Evaluation metrics for multimodal trajectory prediction."""

import torch


def min_ade(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum average displacement error.

    Args:
        predictions: Predicted trajectories with shape [K, T, 2].
        ground_truth: Ground-truth trajectory with shape [T, 2].
    """

    distances = torch.linalg.vector_norm(
        predictions - ground_truth.unsqueeze(0),
        dim=-1,
    )

    ade_per_mode = distances.mean(dim=-1)

    return ade_per_mode.min()


def min_fde(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum final displacement error.

    Args:
        predictions: Predicted trajectories with shape [K, T, 2].
        ground_truth: Ground-truth trajectory with shape [T, 2].
    """

    final_distances = torch.linalg.vector_norm(
        predictions[:, -1] - ground_truth[-1],
        dim=-1,
    )

    return final_distances.min()