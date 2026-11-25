"""Choose a prediction mode using only an observed-motion reference."""
import torch


def choose_history_consistent(trajectories: torch.Tensor, reference: torch.Tensor, steps: int = 10) -> int:
    """Return index closest to a velocity extrapolation over its first `steps` points.

    Both inputs must share coordinates and timestamps. The reference must be
    computed from observed history; future ground truth is not needed.
    """
    if trajectories.ndim != 3 or trajectories.shape[1:] != reference.shape or not len(trajectories):
        raise ValueError('Trajectories and reference must have matching shapes')
    if steps < 1 or not torch.isfinite(trajectories).all() or not torch.isfinite(reference).all():
        raise ValueError('Invalid prediction data')
    n = min(steps, len(reference))
    if n == 0:
        raise ValueError('Trajectories must contain at least one time step')
    distances = torch.linalg.vector_norm(trajectories[:, :n] - reference[:n], dim=-1).mean(dim=-1)
    return int(distances.argmin().item())
