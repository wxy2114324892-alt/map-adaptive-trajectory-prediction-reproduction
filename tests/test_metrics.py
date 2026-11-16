"""Tests for trajectory prediction metrics."""

import torch

from src.utils.metrics import min_ade, min_fde


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Three future positions at the origin.
    ground_truth = torch.zeros(3, 2, device=device)

    predictions = torch.tensor(
        [
            # Good at the first two steps, poor at the final step.
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]],

            # Poor at the first two steps, exact at the final step.
            [[2.0, 0.0], [2.0, 0.0], [0.0, 0.0]],

            # Moderate error at every step.
            [[1.5, 0.0], [1.5, 0.0], [1.5, 0.0]],
        ],
        device=device,
    )

    ade = min_ade(predictions, ground_truth)
    fde = min_fde(predictions, ground_truth)

    assert torch.isclose(ade, torch.tensor(1.0, device=device))
    assert torch.isclose(fde, torch.tensor(0.0, device=device))

    print(f"Device: {device}")
    print(f"minADE: {ade.item():.3f}")
    print(f"minFDE: {fde.item():.3f}")
    print("Metric tests passed.")


if __name__ == "__main__":
    main()