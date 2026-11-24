import unittest
from scripts.analyze_history_quality import categories, summarize


class HistoryAnalysisTests(unittest.TestCase):
    def test_boundaries_and_undefined_angle(self):
        self.assertEqual(categories({'last4_displacement_m': .5, 'short_long_angle_deg': 60}), ('0.5_to_2m', 'at_most_60deg'))
        self.assertEqual(categories({'last4_displacement_m': .1, 'short_long_angle_deg': None}), ('under_0.5m', 'undefined'))
        self.assertEqual(categories({'last4_displacement_m': 2, 'short_long_angle_deg': 61}), ('at_least_2m', 'over_60deg'))

    def test_group_statistics(self):
        rows = [dict(movement='a', direction='x', minADE_m=1., minFDE_m=2., modes=3),
                dict(movement='a', direction='x', minADE_m=3., minFDE_m=4., modes=5)]
        result = summarize(rows)['movement']['a']
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['mean_minADE_m'], 2.)
        self.assertEqual(result['miss_rate_over_2m'], .5)
        self.assertEqual(result['mean_modes'], 4.)
        with self.assertRaises(ValueError): summarize([])


if __name__ == '__main__': unittest.main()
