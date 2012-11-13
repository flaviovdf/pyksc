#-*- coding: utf8
'''Unit tests for the trend module'''

from __future__ import division, print_function

from pyksc.trend import TrendLearner

import unittest
import numpy as np

class TestTrend(unittest.TestCase):

    def addnoise(self, base):
        return np.array(base) + np.random.random(len(base))

    def test_predict_good(self):
        
        base_one = np.ones(10)
        base_two = np.array([90, 2000, 90, 2000, 90, 2000, 90, 2000, 90, 2000])
        
        y = []
        X = []
        for _ in range(10):
            X.append(self.addnoise(base_one))
            X.append(self.addnoise(base_two))
            y.append(1)
            y.append(0)
        
        
        l = TrendLearner(1, 3)
        l.fit(X, y)

        P = []
        for _ in range(50):
            P.append(self.addnoise(base_one))
            P.append(self.addnoise(base_two))

        predict = l.predict(P)
        self.assertEqual(50, sum(predict == 0))
        self.assertEqual(50, sum(predict == 1))

        probs = l.predict_proba(P)
        
        for i in xrange(probs.shape[0]):
            if i % 2 == 0:
                self.assertTrue(probs[i, 1] > probs[i, 0])
            else:
                self.assertTrue(probs[i, 0] > probs[i, 1])
                
    def test_predict_bad(self):
        
        base_one = np.ones(10)
        base_two = np.array([90, 2000, 90, 2000, 90, 2000, 90, 2000, 90, 2000])
        
        y = []
        X = []
        for _ in range(10):
            X.append(self.addnoise(base_one))
            X.append(self.addnoise(base_two))
            y.append(1)
            y.append(0)
        
        
        l = TrendLearner(1, 1)
        l.fit(X, y)

        P = []
        for _ in range(50):
            P.append(self.addnoise(base_one))
            P.append(self.addnoise(base_two))

        predict = l.predict(P)
        self.assertEqual(100, sum(predict == 0))

if __name__ == "__main__":
    unittest.main()