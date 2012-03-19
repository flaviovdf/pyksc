#-*- coding: utf8
'''Unit tests for the dhwt module'''

from __future__ import division, print_function

from math import sqrt
from numpy.testing import *
from pyksc import dhwt

import unittest

import numpy as np

class TestWavelets(unittest.TestCase):

    def test_all(self):
        x = np.array([])
        assert_array_equal(np.array([]), dhwt.transform(x)[0])
        assert_array_equal(np.array([]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))
        
        x = np.array([1., 1])
        assert_array_equal(np.array([1.]), dhwt.transform(x)[0])
        assert_array_equal(np.array([0.]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))
        
        x = np.array([1., 2, 3, 0])
        assert_array_equal(np.array([1.5, 1.5]), dhwt.transform(x)[0])
        assert_array_equal(np.array([-.5, 1.5]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))
        
        x = np.array([1., 2, 3, 0, 7])
        assert_array_equal(np.array([1.5, 1.5, 3.5]), dhwt.transform(x)[0])
        assert_array_equal(np.array([-.5, 1.5, 3.5]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))

        x = np.array([6., 12, 15, 15, 14, 12, 120, 116])
        assert_array_equal(np.array([9., 15, 13, 118]), dhwt.transform(x)[0])
        assert_array_equal(np.array([-3, 0, 1, 2]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))
        
        x = np.array([6., 12, 15, 15, 14, 12, 120, 116, 2])
        assert_array_equal(np.array([9., 15, 13, 118, 1]), dhwt.transform(x)[0])
        assert_array_equal(np.array([-3, 0, 1, 2, 1]), dhwt.transform(x)[1])
        assert_array_equal(x, dhwt.inverse(*dhwt.transform(x)))

if __name__ == "__main__":
    unittest.main()
