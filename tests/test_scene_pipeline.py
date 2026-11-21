"""Checks of variable graph sizes, leakage and saved-model inference."""
import tempfile
import unittest
from pathlib import Path
import torch
import pandas as pd
from src.data.scenes import graph, predict, prepare_scene
from src.models.predictor import MapAdaptiveTrajectoryPredictor


class PipelineTests(unittest.TestCase):
    def test_modes_and_reload(self):
        for owners in ([], [0], [0, 0, 1]):
            scene = {'histories': torch.randn(2, 20, 2), 'centerlines': torch.randn(len(owners), 30, 2), 'owners': owners}
            *_, modes = graph(scene, torch.device('cpu'))
            self.assertEqual(len(modes), owners.count(0) + 2)
            model = MapAdaptiveTrajectoryPredictor(feature_dim=8, hidden_dim=16, heads=2).eval()
            with torch.no_grad(): expected = predict(model, scene, 'cpu')
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.pt'; torch.save(model.state_dict(), path)
                other = MapAdaptiveTrajectoryPredictor(feature_dim=8, hidden_dim=16, heads=2).eval()
                other.load_state_dict(torch.load(path, weights_only=True))
                with torch.no_grad(): actual = predict(other, scene, 'cpu')
            self.assertTrue(torch.allclose(expected, actual))
            self.assertTrue(torch.isfinite(actual).all())

    def test_map_queries_only_receive_history(self):
        class Map:
            def __init__(self): self.calls = []
            def get_candidate_centerlines_for_traj(self, xy, city):
                self.calls.append(xy.copy()); return []
        path = Path('data/raw/forecasting_sample/data/2645.csv')
        if not path.exists(): self.skipTest('Sample data not present')
        amap = Map(); scene = prepare_scene(path, amap)
        self.assertEqual(len(amap.calls), len(scene['histories']))
        self.assertTrue(all(len(x) == 20 for x in amap.calls))
        self.assertEqual(scene['centerlines'].shape[0], 0)

        # Alter every future position: model inputs and map queries must stay fixed.
        data = pd.read_csv(path)
        timestamps = sorted(data.TIMESTAMP.unique())
        data.loc[data.TIMESTAMP.isin(timestamps[20:]), ['X', 'Y']] += 1000
        with tempfile.TemporaryDirectory() as tmp:
            modified = Path(tmp) / 'scene.csv'
            data.to_csv(modified, index=False)
            other_map = Map()
            other = prepare_scene(modified, other_map)
            robust_map = Map()
            robust_other = prepare_scene(modified, robust_map, heading_strategy='robust')
        robust_original_map = Map()
        robust_original = prepare_scene(path, robust_original_map, heading_strategy='robust')
        self.assertTrue(torch.equal(scene['histories'], other['histories']))
        self.assertTrue(all((a == b).all() for a, b in zip(amap.calls, other_map.calls)))
        self.assertFalse(torch.equal(scene['future'], other['future']))
        self.assertTrue(torch.equal(robust_original['histories'], robust_other['histories']))
        self.assertEqual(robust_original['history_quality'], robust_other['history_quality'])
        self.assertTrue(all((a == b).all() for a, b in zip(robust_original_map.calls, robust_map.calls)))

    def test_missing_target_history_rejected(self):
        path = Path('data/raw/forecasting_sample/data/2645.csv')
        if not path.exists(): self.skipTest('Sample data not present')
        data = pd.read_csv(path)
        first = data.loc[data.OBJECT_TYPE == 'AGENT'].sort_values('TIMESTAMP').index[0]
        with tempfile.TemporaryDirectory() as tmp:
            modified = Path(tmp) / 'scene.csv'
            data.drop(index=first).to_csv(modified, index=False)
            with self.assertRaises(ValueError):
                prepare_scene(modified, None)


if __name__ == '__main__': unittest.main()
