import unittest
import torch
from src.models.candidate_scorer import candidate_features, CandidateScorer, select_candidate
from scripts.train_candidate_scorer import build_record, batch_loss, make_splits


class CandidateScorerTests(unittest.TestCase):
    def test_feature_shapes_types_and_no_future_input(self):
        history=torch.zeros(20,2); predictions=torch.zeros(4,30,2); reference=torch.zeros(30,2)
        features=candidate_features(history,predictions,reference)
        self.assertEqual(features.shape,(4,163))
        self.assertTrue(torch.equal(features[:,-3:],torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])))
        self.assertEqual(CandidateScorer()(features).shape,(4,))
        chosen=select_candidate(CandidateScorer(),history,predictions,reference)
        self.assertTrue(0 <= chosen < 4)
        with self.assertRaises(ValueError): candidate_features(history,predictions[:1],reference)

    def test_mode_permutation_equivariance(self):
        net=CandidateScorer().eval(); features=torch.randn(5,163)
        order=torch.tensor([3,0,4,1,2])
        self.assertTrue(torch.allclose(net(features)[order],net(features[order])))

    def test_future_changes_labels_not_input_features(self):
        history=torch.stack((torch.arange(20).float(), torch.zeros(20)), dim=-1)
        predictions=torch.stack((torch.zeros(30,2), torch.ones(30,2)))
        scene={'histories':history.unsqueeze(0), 'intervals':torch.full((19,),.1), 'future':torch.zeros(30,2)}
        first=build_record(scene,predictions)
        scene['future']=torch.ones(30,2)
        second=build_record(scene,predictions)
        self.assertTrue(torch.equal(first['features'],second['features']))
        self.assertNotEqual(first['ade'].argmin().item(),second['ade'].argmin().item())
        scene['future'][0,0]=float('nan')
        with self.assertRaises(ValueError): build_record(scene,predictions)

    def test_batch_padding_does_not_change_single_scene_loss(self):
        model=CandidateScorer().eval()
        one={'features':torch.randn(2,163),'ade':torch.tensor([1.,2.])}
        other={'features':torch.randn(5,163),'ade':torch.tensor([1.,2.,3.,4.,5.])}
        single=batch_loss(model,[one],'cpu')
        mixed=batch_loss(model,[one,other],'cpu')
        second=batch_loss(model,[other],'cpu')
        self.assertTrue(torch.allclose(mixed,(single+second)/2,atol=1e-6))

    def test_reproducible_disjoint_training_and_tuning(self):
        train=[str(i) for i in range(20)]; val=['v1','v2','v3']
        parts=make_splits(train,val,10,5,0,42)
        self.assertEqual(parts,make_splits(list(reversed(train)),val,10,5,0,42))
        self.assertFalse(set(parts['train'][1]) & set(parts['tune'][1]))
        self.assertEqual(set(parts['val'][1]),set(val))
        with self.assertRaises(ValueError): make_splits(train,val,10,11,0,42)


if __name__=='__main__': unittest.main()
