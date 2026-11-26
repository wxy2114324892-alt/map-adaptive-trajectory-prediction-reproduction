"""Input-only features and an experimental per-trajectory scoring network."""
import torch
from torch import nn


def candidate_features(history, predictions, reference):
    """Build per-mode features without the recorded future or any labels.

    All positions are in the same target-centered frame, in meters. Output
    order is target centerlines, scene reasoning, then motion maintaining.
    """
    if history.shape != (20, 2) or predictions.ndim != 3 or predictions.shape[1:] != (30, 2):
        raise ValueError('Expected history [20,2] and predictions [K,30,2]')
    if predictions.shape[0] < 2 or reference.shape != (30, 2):
        raise ValueError('Expected at least two modes and a [30,2] reference')
    if not all(torch.isfinite(x).all() for x in (history, predictions, reference)):
        raise ValueError('Inputs must be finite')
    k = len(predictions)
    kinds = predictions.new_zeros((k, 3))
    kinds[:-2, 0] = 1
    kinds[-2, 1] = 1
    kinds[-1, 2] = 1
    return torch.cat((predictions.reshape(k, -1)/50,
                      (predictions-reference).reshape(k, -1)/50,
                      history.reshape(1, -1).expand(k, -1)/50,
                      kinds), dim=1)


class CandidateScorer(nn.Module):
    """Score each mode independently; higher means more likely best-ADE mode."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(163, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def select_candidate(scorer, history, predictions, reference):
    """Select a candidate from observed inputs; no future label is required."""
    with torch.no_grad():
        features = candidate_features(history, predictions, reference)
        return int(scorer(features).argmax().item())
