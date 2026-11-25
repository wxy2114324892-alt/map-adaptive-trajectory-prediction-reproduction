import unittest
import torch
from scripts.evaluate_prediction_branches import trajectory_errors, summarize, choose_history_consistent


class BranchComparisonTests(unittest.TestCase):
    def test_branch_order_and_independent_oracles(self):
        truth = torch.zeros(2, 2)
        predicted = torch.tensor([[[0., 0.], [2., 0.]], [[3., 0.], [0., 0.]], [[2., 0.], [2., 0.]]])
        scores = trajectory_errors(predicted, truth)
        self.assertEqual(scores['scene'], (1.5, 0.))
        self.assertEqual(scores['motion'], (2., 2.))
        self.assertEqual(scores['oracle'], (1., 0.))
        with self.assertRaises(ValueError): trajectory_errors(predicted[:1], truth)

    def test_aggregate_miss_is_strict(self):
        rows = [{'constant_velocity': (1.,2.), 'scene': (2.,3.), 'motion': (3.,4.), 'history_match': (1.,3.), 'oracle': (1.,1.)}]
        scores = summarize(rows)
        self.assertEqual(scores['constant_velocity']['miss_rate_over_2m'], 0.)
        self.assertEqual(scores['scene']['miss_rate_over_2m'], 1.)
        with self.assertRaises(ValueError): summarize([])

    def test_selection_uses_only_reference_and_predictions(self):
        reference = torch.tensor([[1., 0.], [2., 0.]])
        predictions = torch.tensor([[[1., 1.], [2., 1.]], [[1., 0.], [2., 0.]]])
        self.assertEqual(choose_history_consistent(predictions, reference), 1)
        with self.assertRaises(ValueError): choose_history_consistent(predictions, reference, steps=0)
        with self.assertRaises(ValueError): choose_history_consistent(torch.empty(2, 0, 2), torch.empty(0, 2))


if __name__=='__main__': unittest.main()
