"""Test for the modified MTP loss."""

import torch

from src.models.losses import modified_mtp_loss


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ground truth: a stationary trajectory at the origin.
    ground_truth = torch.zeros(30, 2, device=device)

    # Four predictions with different distances from the ground truth.
    predictions = torch.stack(
        [
            torch.full((30, 2), 3.0),
            torch.full((30, 2), 1.0),
            torch.full((30, 2), 0.1),
            torch.full((30, 2), -2.0),
        ],
        dim=0,
    ).to(device)

    # Enable gradient calculation for the predictions.
    predictions.requires_grad_()

    loss, best_mode = modified_mtp_loss(
        predictions,
        ground_truth,
    )

    # Mode 2 is closest to the ground truth.
    assert best_mode.item() == 2

    # Check that backpropagation works.
    loss.backward()
    assert predictions.grad is not None

    # Only the selected mode should receive regression gradients.
    gradient_per_mode = predictions.grad.abs().sum(dim=(1, 2))

    assert gradient_per_mode[2] > 0
    assert gradient_per_mode[0] == 0
    assert gradient_per_mode[1] == 0
    assert gradient_per_mode[3] == 0

    print(f"Device: {device}")
    print(f"Best mode: {best_mode.item()}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient per mode: {gradient_per_mode.tolist()}")
    print("Modified MTP loss test passed.")


if __name__ == "__main__":
    main()