"""Checks for validation-error summaries and endpoint threshold behavior."""
import unittest
from scripts.analyze_validation_errors import summarize


class SummaryTests(unittest.TestCase):
    def test_percentiles_and_strict_miss_threshold(self):
        report = {'val_count': 3, 'scenes': [
            {'minADE_m': 1.0, 'minFDE_m': 1.0, 'modes': 2},
            {'minADE_m': 2.0, 'minFDE_m': 2.0, 'modes': 3},
            {'minADE_m': 3.0, 'minFDE_m': 3.0, 'modes': 5},
        ]}
        result = summarize(report, 2.0)
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['minADE_m']['median'], 2.0)
        self.assertAlmostEqual(result['endpoint_miss_rate'], 1/3)
        self.assertEqual((result['modes_min'], result['modes_max']), (2, 5))

    def test_rejects_invalid_or_incomplete_results(self):
        with self.assertRaises(ValueError):
            summarize({'val_count': 2, 'scenes': [{'minADE_m': 1, 'minFDE_m': 1, 'modes': 2}]})
        with self.assertRaises(ValueError):
            summarize({'val_count': 1, 'scenes': [{'minADE_m': float('nan'), 'minFDE_m': 1, 'modes': 2}]})


if __name__ == '__main__':
    unittest.main()
