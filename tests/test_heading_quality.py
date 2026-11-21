import unittest
import sqlite3
import torch
from src.data.argoverse_loader import estimate_heading, align_to_target_heading, history_quality
from scripts.prepare_dataset import configure_heading_strategy


class HeadingQualityTests(unittest.TestCase):
    def test_legacy_and_robust_agree_on_straight_motion(self):
        history = torch.stack((torch.arange(20).float(), torch.zeros(20)), dim=1)
        for strategy in ('legacy', 'robust'):
            angle, source = estimate_heading(history, strategy)
            self.assertAlmostEqual(angle.item(), 0.0)
            self.assertEqual(source, 'short')
            self.assertTrue(torch.allclose(align_to_target_heading(history, history, strategy), history))

    def test_robust_falls_back_when_recent_points_repeat(self):
        history = torch.stack((torch.zeros(20), torch.arange(20).float()), dim=1)
        history[-5:] = history[-5]
        angle, source = estimate_heading(history, 'robust')
        self.assertEqual(source, 'long_small_displacement')
        self.assertAlmostEqual(angle.item(), torch.pi / 2, places=5)
        self.assertTrue(history_quality(history, torch.full((19,), 0.1))['last5_repeated_1cm'])

    def test_disagreement_and_stationary_fallback(self):
        history = torch.zeros(20, 2)
        history[-11] = torch.tensor([0., -10.])
        history[-5] = torch.tensor([-2., 0.])
        angle, source = estimate_heading(history, 'robust')
        self.assertEqual(source, 'long_disagreement')
        self.assertAlmostEqual(angle.item(), torch.pi / 2, places=5)
        angle, source = estimate_heading(torch.zeros(20, 2), 'robust')
        self.assertEqual(source, 'stationary_fallback')
        self.assertEqual(angle.item(), 0.)

    def test_invalid_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            history_quality(torch.zeros(20, 2), torch.zeros(19))

    def test_cache_strategy_preserves_existing_and_rejects_mixing(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE scenes (split TEXT, id TEXT, payload BLOB)')
            configure_heading_strategy(db, 'robust')
            configure_heading_strategy(db, 'robust')
            with self.assertRaisesRegex(ValueError, 'separate cache'):
                configure_heading_strategy(db, 'legacy')
            self.assertEqual(db.execute('SELECT value FROM cache_metadata').fetchone()[0], 'robust')

        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE scenes (split TEXT, id TEXT, payload BLOB)')
            db.execute("INSERT INTO scenes VALUES ('train', 'old', X'00')")
            db.commit()
            with self.assertRaisesRegex(ValueError, 'uses legacy'):
                configure_heading_strategy(db, 'robust')
            configure_heading_strategy(db, 'legacy')
            self.assertEqual(db.execute('SELECT value FROM cache_metadata').fetchone()[0], 'legacy')


if __name__ == '__main__':
    unittest.main()
