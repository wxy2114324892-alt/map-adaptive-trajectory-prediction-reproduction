"""Loss functions for multimodal trajectory prediction."""

import torch
from torch.nn import functional as F


def modified_mtp_loss(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute regression loss for the prediction closest to ground truth.

    Args:
        predictions: Predicted trajectories with shape [K, T, 2].
        ground_truth: Ground-truth trajectory with shape [T, 2].

    Returns:
        Smooth L1 loss and the index of the best prediction mode.
    """

    # Compare every prediction with the same ground-truth trajectory.
    displacement = predictions - ground_truth.unsqueeze(0)

    # Compute the average displacement error for each mode.
    ade_per_mode = torch.linalg.vector_norm(
        displacement,
        dim=-1,
    ).mean(dim=-1)

    # Select the prediction closest to the ground truth.
    best_mode = torch.argmin(ade_per_mode)

    # Optimize only the selected prediction.
    loss = F.smooth_l1_loss(
        predictions[best_mode],
        ground_truth,
    )

    return loss, best_mode