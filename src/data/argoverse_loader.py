"""Utilities for loading Argoverse 1 forecasting scenarios."""

from pathlib import Path

import pandas as pd
import torch


def load_agent_trajectory(
    csv_path: str | Path,
    history_steps: int = 20,
    future_steps: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the target agent trajectory.

    Returns:
        history: Target history with shape [history_steps, 2].
        future: Ground-truth future with shape [future_steps, 2].
        origin: Global position of the last history point.
    """

    dataframe = pd.read_csv(csv_path)

    agent_rows = dataframe[
        dataframe["OBJECT_TYPE"] == "AGENT"
    ].sort_values("TIMESTAMP")

    required_steps = history_steps + future_steps

    if len(agent_rows) < required_steps:
        raise ValueError(
            f"Expected at least {required_steps} AGENT rows, "
            f"but found {len(agent_rows)}."
        )

    coordinates = torch.tensor(
        agent_rows[["X", "Y"]]
        .iloc[:required_steps]
        .to_numpy(),
        dtype=torch.float64,
    )

    # Use the final observed position as the local origin.
    origin = coordinates[history_steps - 1].clone()
    relative_coordinates = coordinates - origin

    history = relative_coordinates[:history_steps].float()
    future = relative_coordinates[
        history_steps : history_steps + future_steps
    ].float()

    return history, future, origin


def load_scene_histories(
    csv_path: str | Path,
    history_steps: int = 20,
    max_distance: float = 50.0,
) -> tuple[torch.Tensor, list[str], list[str], torch.Tensor]:
    """
    Load the target and surrounding-agent histories.

    Returns:
        histories: Relative trajectories with shape [num_agents, T, 2].
        track_ids: Track ID corresponding to each trajectory.
        object_types: Object type corresponding to each trajectory.
        origin: Global position of the target at the current time.
    """

    dataframe = pd.read_csv(csv_path)
    timestamps = sorted(dataframe["TIMESTAMP"].unique())

    if len(timestamps) < history_steps:
        raise ValueError(
            f"Expected at least {history_steps} timestamps, "
            f"but found {len(timestamps)}."
        )

    history_timestamps = timestamps[:history_steps]

    target_rows = (
        dataframe[dataframe["OBJECT_TYPE"] == "AGENT"]
        .sort_values("TIMESTAMP")
        .iloc[:history_steps]
    )

    if len(target_rows) < history_steps:
        raise ValueError("Target agent does not have a complete history.")

    target_track_id = target_rows["TRACK_ID"].iloc[0]

    origin = torch.tensor(
        target_rows[["X", "Y"]].iloc[-1].to_numpy(),
        dtype=torch.float64,
    )

    # Put the target agent before all surrounding agents.
    ordered_track_ids = [target_track_id]

    ordered_track_ids.extend(
        track_id
        for track_id in dataframe["TRACK_ID"].unique()
        if track_id != target_track_id
    )

    histories = []
    track_ids = []
    object_types = []

    # Keep tracks visible during the complete history period.
    for track_id in ordered_track_ids:
        track_rows = (
            dataframe[
                (dataframe["TRACK_ID"] == track_id)
                & (dataframe["TIMESTAMP"].isin(history_timestamps))
            ]
            .sort_values("TIMESTAMP")
        )

        if len(track_rows) != history_steps:
            continue

        coordinates = torch.tensor(
            track_rows[["X", "Y"]].to_numpy(),
            dtype=torch.float64,
        )

        relative_coordinates = (coordinates - origin).float()
        current_distance = torch.linalg.vector_norm(
            relative_coordinates[-1]
        )

        # Always keep the target and discard distant agents.
        is_target = track_id == target_track_id

        if not is_target and current_distance > max_distance:
            continue

        histories.append(relative_coordinates)
        track_ids.append(str(track_id))
        object_types.append(str(track_rows["OBJECT_TYPE"].iloc[0]))

    if not histories:
        raise ValueError("No complete agent histories were found.")

    histories_tensor = torch.stack(histories)

    return histories_tensor, track_ids, object_types, origin

def estimate_heading(target_history: torch.Tensor, strategy: str = "legacy") -> tuple[torch.Tensor, str]:
    """Estimate heading from observed positions only; robust mode is opt-in."""
    if target_history.shape[0] < 11 and strategy == "robust":
        raise ValueError("Robust heading requires at least eleven history points.")
    if target_history.shape[0] < 5:
        raise ValueError("At least five history points are required.")
    if strategy not in ("legacy", "robust"):
        raise ValueError(f"Unknown heading strategy: {strategy}")
    short = target_history[-1] - target_history[-5]
    if strategy == "legacy":
        return torch.atan2(short[1], short[0]), "short"
    long = target_history[-1] - target_history[-11]
    short_norm = torch.linalg.vector_norm(short)
    long_norm = torch.linalg.vector_norm(long)
    if short_norm >= 0.5:
        if long_norm >= 0.5:
            cosine = torch.dot(short, long) / (short_norm * long_norm)
            if cosine < 0.5:  # More than 60 degrees apart.
                return torch.atan2(long[1], long[0]), "long_disagreement"
        return torch.atan2(short[1], short[0]), "short"
    if long_norm >= 0.5:
        return torch.atan2(long[1], long[0]), "long_small_displacement"
    return short.new_zeros(()), "stationary_fallback"


def history_quality(target_history: torch.Tensor, intervals: torch.Tensor) -> dict:
    """Return screening indicators; flags are not automatic data rejection."""
    steps = target_history[1:] - target_history[:-1]
    if len(intervals) != len(steps) or torch.any(intervals <= 0):
        raise ValueError("Positive intervals are required for each history step.")
    short = target_history[-1] - target_history[-5]
    long = target_history[-1] - target_history[-11]
    short_norm = torch.linalg.vector_norm(short)
    long_norm = torch.linalg.vector_norm(long)
    angle = None
    if short_norm > 0.1 and long_norm > 0.1:
        cosine = (torch.dot(short, long) / (short_norm * long_norm)).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cosine)).item()
    return {
        "last4_displacement_m": short_norm.item(),
        "max_step_speed_mps": (torch.linalg.vector_norm(steps, dim=-1) / intervals).max().item(),
        "last5_repeated_1cm": bool(torch.all(torch.linalg.vector_norm(target_history[-5:] - target_history[-1], dim=-1) < 0.01)),
        "short_long_angle_deg": angle,
    }


def align_to_target_heading(
    coordinates: torch.Tensor,
    target_history: torch.Tensor,
    strategy: str = "legacy",
) -> torch.Tensor:
    """Rotate coordinates using an observed-history heading estimate."""
    heading_angle, _ = estimate_heading(target_history, strategy)
    cosine = torch.cos(heading_angle)
    sine = torch.sin(heading_angle)
    rotation_matrix = torch.stack(
        [torch.stack([cosine, -sine]), torch.stack([sine, cosine])]
    )
    return coordinates @ rotation_matrix
