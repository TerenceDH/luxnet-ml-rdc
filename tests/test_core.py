"""Checks for the physical optimization contract and checkpoint portability."""
import unittest
import numpy as np
import torch
from luxnet.optimization import evaluate_population,pso_optimize_lighting_fast
from luxnet.model import load_model

class CoreTests(unittest.TestCase):
    def test_all_off_is_not_false_zero_objective(self):
        val,*_=evaluate_population(np.zeros(2),np.ones((1,2)),np.zeros((1,1)),500)
        self.assertEqual(float(val[0]),250000)

    def test_optimize_solvable_single_lamp(self):
        r=pso_optimize_lighting_fast(np.zeros(4),np.full((1,4),1000),500)
        self.assertLess(abs(float(r['best_ratios'][0])-.5),.01)
        self.assertTrue(np.all(np.diff(r['history'])<=1e-4))

    def test_vector_target(self):
        target=np.array([300,800],dtype=np.float32)
        val,*_=evaluate_population(np.zeros(2),np.diag(target),np.ones((1,2)),target)
        self.assertEqual(float(val[0]),0)

    def test_checkpoint(self):
        model,mean,std=load_model('weights/mresunet_epoch888.pt')
        self.assertEqual(mean.shape,(11,));self.assertTrue((std>0).all())
        with torch.inference_mode(): y=model(torch.zeros(1,3,128,128),torch.zeros(1,11))
        self.assertEqual(tuple(y.shape),(1,1,128,128))
        self.assertTrue(torch.isfinite(y).all());self.assertTrue((y>=0).all())

if __name__=='__main__':unittest.main()
