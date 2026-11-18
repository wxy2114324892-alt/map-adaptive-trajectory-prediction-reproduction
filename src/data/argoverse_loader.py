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

def align_to_target_heading(
    coordinates: torch.Tensor,
    target_history: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate coordinates so the target heading points along positive X.
    """

    if target_history.shape[0] < 5:
        raise ValueError("At least five history points are required.")

    # Estimate heading from the latest five history points.
    heading_vector = target_history[-1] - target_history[-5]

    heading_angle = torch.atan2(
        heading_vector[1],
        heading_vector[0],
    )

    cosine = torch.cos(heading_angle)
    sine = torch.sin(heading_angle)

    rotation_matrix = torch.stack(
        [
            torch.stack([cosine, -sine]),
            torch.stack([sine, cosine]),
        ]
    )

    return coordinates @ rotation_matrix
