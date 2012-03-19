#-*- coding: utf8
'''Unit tests for the ksc module'''

from __future__ import division, print_function

from pyksc import dist
from pyksc import ksc

import unittest

import numpy as np

class TestKSC(unittest.TestCase):
    
    def ksc_runner(self, method):
        k = 2
        #One cluster with uniform series, another with a peak.
        X = np.array([[1.0,1,1],
                      [1.1,1,1],
                      [1.2,1,1],
                      [1.3,1,1],
                      [1.3,1,1],
                      [1.3,1,1],
                      [1.3,1,1],
                      [1.3,1,1],
                      [90,2000,90],
                      [90,2001,90],
                      [90,2002,90],
                      [90,2003,90]])
        
        cents, assign, shift, distc = ksc.ksc(X, k)
        del shift
        
        self.assertEqual(len(set(assign)), k)
        self.assertEqual(sum(assign == assign[0]), 8)
        self.assertEqual(sum(assign == assign[-1]), 4)
        
        self.assertEqual(len(set(assign[:8])), 1)
        self.assertEqual(len(set(assign[8:])), 1)
        self.assertFalse(set(assign[:8]) == set(assign[8:]))

        cluster_one = assign[0]
        cluster_two = assign[-1]
        
        self.assertTrue(dist.dist(X[0], cents[cluster_one]) < \
                        dist.dist(X[0], cents[cluster_two]))
        self.assertTrue(dist.dist(cents[cluster_one], cents[cluster_two]) > 0)
        
        for i in xrange(X.shape[0]):
            self.assertAlmostEqual(dist.dist(X[i], cents[0], True), 
                                   distc.T[i, 0], 5)
            self.assertAlmostEqual(dist.dist(X[i], cents[1], True), 
                                   distc.T[i, 1], 5)
        
    def test_clustering(self):
        self.ksc_runner(ksc.ksc)
        
    def test_incremental_cluster(self):
        self.ksc_runner(ksc.inc_ksc)
    
if __name__ == "__main__":
    unittest.main()