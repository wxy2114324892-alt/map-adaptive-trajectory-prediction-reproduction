import unittest
import torch
from scripts.compare_constant_velocity import constant_velocity, summarize


class ConstantVelocityTests(unittest.TestCase):
    def test_exact_linear_motion_and_stationary(self):
        history = torch.tensor([[float(i), 2.*i] for i in range(5)])
        expected = torch.tensor([[5., 10.], [6., 12.]])
        self.assertTrue(torch.allclose(constant_velocity(history, torch.ones(4)*.1, 2), expected))
        self.assertTrue(torch.equal(constant_velocity(torch.zeros(5, 2), torch.ones(4)*.1, 3), torch.zeros(3, 2)))

    def test_irregular_observation_intervals_and_invalid_data(self):
        history = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        self.assertTrue(torch.allclose(constant_velocity(history, torch.tensor([.05,.1,.1,.15]), 1), torch.tensor([[5., 0.]])))
        with self.assertRaises(ValueError): constant_velocity(history, torch.tensor([.1,.1,0.,.1]), 1)

    def test_metrics_use_strict_miss_threshold(self):
        rows = [{'baseline_ade_m': 1., 'baseline_fde_m': 2., 'model_minADE_m': 2., 'model_minFDE_m': 1.},
                {'baseline_ade_m': 3., 'baseline_fde_m': 4., 'model_minADE_m': 1., 'model_minFDE_m': 3.}]
        result = summarize(rows)
        self.assertEqual(result['baseline_fde_m']['mean'], 3.)
        self.assertEqual(result['baseline_miss_rate_over_2m'], .5)
        self.assertEqual(result['model_lower_ADE_fraction'], .5)


if __name__ == '__main__': unittest.main()
