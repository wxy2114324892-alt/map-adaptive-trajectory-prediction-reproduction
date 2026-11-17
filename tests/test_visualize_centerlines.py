"""Tests for the candidate-centerline visualization command.

Uses synthetic observations and a fake map API, so no dataset, map or cloned
Argoverse API is needed. Run with: python -m unittest tests.test_visualize_centerlines
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import visualize_centerlines


class FakeMap:
    instances = []
    result = [np.array([[18.0, 0.0], [19.0, 1.0]])]

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.__class__.instances.append(self)

    def get_candidate_centerlines_for_traj(self, history, city):
        self.calls.append((history.copy(), city))
        return self.result


class TestCenterlineVisualization(unittest.TestCase):
    def setUp(self):
        FakeMap.instances = []
        FakeMap.result = [np.array([[18.0, 0.0], [19.0, 1.0]])]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.csv = self.folder / "scene.csv"
        self.output = self.folder / "nested" / "plot.png"
        self.map_dir = self.folder / "maps"
        module = types.ModuleType("argoverse.map_representation.map_api")
        module.ArgoverseMap = FakeMap
        self.module_patch = patch.dict(sys.modules, {
            "argoverse": types.ModuleType("argoverse"),
            "argoverse.map_representation": types.ModuleType("argoverse.map_representation"),
            "argoverse.map_representation.map_api": module,
        })
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def write_scene(self, steps=50):
        # Reverse rows to verify timestamp sorting. Future positions are distinct.
        rows = [{"TIMESTAMP": t, "OBJECT_TYPE": "AGENT", "X": float(t),
                 "Y": float(t * 2), "CITY_NAME": "MIA"}
                for t in range(steps)]
        rows.append({"TIMESTAMP": 0, "OBJECT_TYPE": "OTHERS", "X": -99,
                     "Y": -99, "CITY_NAME": "MIA"})
        pd.DataFrame(rows[::-1]).to_csv(self.csv, index=False)

    def run_command(self):
        args = ["visualize_centerlines.py", "--scene", str(self.csv),
                "--map-dir", str(self.map_dir), "--output", str(self.output)]
        with patch.object(sys, "argv", args):
            visualize_centerlines.main()

    def test_map_receives_only_sorted_history_and_image_is_written(self):
        self.write_scene()
        self.run_command()
        self.assertTrue(self.output.is_file())
        self.assertGreater(self.output.stat().st_size, 0)
        self.assertEqual(FakeMap.instances[0].root, self.map_dir)
        self.assertEqual(len(FakeMap.instances[0].calls), 1)
        history, city = FakeMap.instances[0].calls[0]
        np.testing.assert_array_equal(history, [[float(t), float(t * 2)] for t in range(20)])
        self.assertEqual(city, "MIA")

    def test_short_history_fails_before_map_query(self):
        self.write_scene(19)
        with self.assertRaisesRegex(ValueError, "at least 20"):
            self.run_command()
        self.assertEqual(FakeMap.instances, [])
        self.assertFalse(self.output.exists())

    def test_empty_candidates_fails_clearly(self):
        self.write_scene()
        FakeMap.result = []
        with self.assertRaisesRegex(RuntimeError, "No candidate centerlines"):
            self.run_command()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
