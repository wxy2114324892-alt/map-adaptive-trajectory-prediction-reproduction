import unittest
from scripts.analyze_selector_groups import compare_groups


class SelectorGroupTests(unittest.TestCase):
    def setUp(self):
        self.selector={'validation_count':1,'scenes':[{'id':'a','constant_velocity':[2.,3.], 'heuristic':[1.,3.], 'learned':[1.,2.], 'oracle':[.5,1.]}]}
        self.baseline={'overall':{'count':1},'rows':[{'id':'a','movement':'slow','baseline_ade_m':2.,'baseline_fde_m':3.}]}

    def test_group_metrics_and_baseline_match(self):
        result=compare_groups(self.selector,self.baseline)['groups']['slow']
        self.assertEqual(result['count'],1)
        self.assertEqual(result['methods']['learned']['mean_FDE_m'],2.)
        self.assertEqual(result['learned_better_FDE_than_baseline'],1.)

    def test_mismatched_scene_or_baseline_rejected(self):
        self.baseline['rows'][0]['id']='b'
        with self.assertRaises(ValueError): compare_groups(self.selector,self.baseline)
        self.baseline['rows'][0]['id']='a'; self.baseline['rows'][0]['baseline_fde_m']=8.
        with self.assertRaises(ValueError): compare_groups(self.selector,self.baseline)


if __name__=='__main__': unittest.main()
