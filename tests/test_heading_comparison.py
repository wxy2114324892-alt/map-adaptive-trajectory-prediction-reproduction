import unittest
import sqlite3
import torch
from scripts.prepare_heading_comparison import robust_from_legacy, verify_legacy_source
from src.data.argoverse_loader import align_to_target_heading


class PairedHeadingTests(unittest.TestCase):
    def test_rotation_matches_direct_robust_transform(self):
        history = torch.zeros(20, 2)
        history[-11] = torch.tensor([0., -10.])
        history[-5] = torch.tensor([-2., 0.])
        future = torch.tensor([[2., 3.], [3., 5.]])
        lines = torch.tensor([[[1., 2.], [3., 4.]]])
        scene = {'histories': align_to_target_heading(history.unsqueeze(0), history),
                 'future': align_to_target_heading(future, history),
                 'centerlines': align_to_target_heading(lines, history)}
        result = robust_from_legacy(scene)
        self.assertEqual(result['heading_source'], 'long_disagreement')
        self.assertTrue(torch.allclose(result['future'], align_to_target_heading(future, history, 'robust'), atol=1e-5))
        self.assertTrue(torch.allclose(result['centerlines'], align_to_target_heading(lines, history, 'robust'), atol=1e-5))
        self.assertTrue(torch.allclose(result['histories'][0], align_to_target_heading(history, history, 'robust'), atol=1e-5))

    def test_stationary_global_fallback_cannot_be_recovered(self):
        self.assertIsNone(robust_from_legacy({'histories': torch.zeros(1, 20, 2)}))

    def test_source_strategy_must_not_be_robust(self):
        with sqlite3.connect(':memory:') as db:
            verify_legacy_source(db)  # Old caches may not have metadata.
            db.execute('CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT)')
            db.execute("INSERT INTO cache_metadata VALUES ('heading_strategy', 'robust')")
            with self.assertRaisesRegex(ValueError, 'legacy source cache'):
                verify_legacy_source(db)


if __name__ == '__main__': unittest.main()
