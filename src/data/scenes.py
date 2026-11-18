"""Prepare Argoverse scenes without using future positions for map queries."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data.argoverse_loader import load_agent_trajectory, load_scene_histories, align_to_target_heading


def resample(line, count=30):
    line = np.asarray(line, dtype=np.float64)[:, :2]
    if len(line) < 2:
        raise ValueError('Centerline must contain at least two points')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    distance, line = distance[keep], line[keep]
    if len(line) < 2:
        raise ValueError('Degenerate centerline')
    samples = np.linspace(0, distance[-1], count)
    return np.stack([np.interp(samples, distance, line[:, i]) for i in range(2)], axis=1)


def prepare_scene(path, amap):
    path = Path(path)
    history, future, origin = load_agent_trajectory(path)
    histories, ids, types, scene_origin = load_scene_histories(path)
    if not torch.allclose(origin, scene_origin) or not torch.allclose(history, histories[0]):
        raise ValueError(f'{path}: inconsistent target history or timestamps')
    data = pd.read_csv(path)
    timestamps = np.sort(data['TIMESTAMP'].unique())[:20]
    intervals = np.diff(timestamps)
    if len(timestamps) != 20 or np.any(intervals <= 0) or np.any(intervals > 0.5):
        raise ValueError(f'{path}: invalid observation timestamps or gap above 0.5 seconds')
    city = data.loc[data.OBJECT_TYPE == 'AGENT', 'CITY_NAME'].iloc[0]
    lines, owners = [], []
    for owner, track in enumerate(histories):
        # The map API receives observed global positions only.
        candidates = amap.get_candidate_centerlines_for_traj((track + origin).numpy(), city)
        for candidate in candidates:
            try:
                line = (torch.tensor(resample(candidate), dtype=torch.float64) - origin).float()
            except ValueError:
                continue
            lines.append(align_to_target_heading(line, history))
            owners.append(owner)
    return {'id': path.stem, 'histories': align_to_target_heading(histories, history),
            'future': align_to_target_heading(future, history),
            'centerlines': torch.stack(lines) if lines else torch.empty(0, 30, 2),
            'owners': owners, 'origin': origin, 'city': city, 'track_ids': ids,
            'intervals': torch.tensor(intervals, dtype=torch.float32)}


def graph(scene, device):
    n = len(scene['histories']); owners = scene['owners']; total = n + len(owners) + 1
    loops = [(i, i) for i in range(total)]
    edge = lambda pairs: torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
    target_lines = [n + i for i, owner in enumerate(owners) if owner == 0]
    maintained = [0] + target_lines + [total - 1]
    preserved = [(i, i) for i in maintained]
    stage1 = [(n+i, owner) for i, owner in enumerate(owners) if owner != 0] + [(i, i) for i in range(n)] + [(i, i) for i in target_lines] + [(total-1, total-1)]
    stage2 = [(i, 0) for i in range(1, n)] + [(i, i) for i in range(n)] + [(i, i) for i in target_lines] + [(total-1, total-1)]
    stage3 = [(i, 0) for i in target_lines] + preserved
    prediction = [(0, i) for i in target_lines] + [(i, 0) for i in target_lines] + preserved
    modes = torch.tensor(target_lines + [0, total-1], dtype=torch.long, device=device)
    return *(edge(x) for x in (stage1, stage2, stage3, prediction)), modes


def predict(model, scene, device):
    history = scene['histories'].to(device)
    if model.agent_encoder.gru.input_size == 4:
        velocity = torch.zeros_like(history)
        intervals = scene.get('intervals', torch.full((history.shape[1]-1,), 0.1)).to(device)
        velocity[:, 1:] = (history[:, 1:] - history[:, :-1]) / intervals.view(1, -1, 1)
        velocity[:, 0] = velocity[:, 1]
        history = torch.cat((history, velocity), dim=-1)
    return model(history, scene['centerlines'].to(device), *graph(scene, device))
