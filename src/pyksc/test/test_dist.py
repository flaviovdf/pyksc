#-*- coding: utf8
'''Unit tests for the dist module'''

from __future__ import division, print_function

from math import sqrt
from numpy.testing import *
from pyksc import dist

import unittest

import numpy as np

class TestDist(unittest.TestCase):

    def test_shift_roll(self):
        array = np.array([])
        assert_array_equal(np.array([]), dist.shift(array, 0))
        assert_array_equal(np.array([]), dist.shift(array, -1))
        assert_array_equal(np.array([]), dist.shift(array, 1))
        assert_array_equal(np.array([]), dist.shift(array, 10))
        assert_array_equal(np.array([]), dist.shift(array, -10))

        array = np.array([1.0])
        assert_array_equal(np.array([1.0]), dist.shift(array, 0, True))
        assert_array_equal(np.array([1.0]), dist.shift(array, 1, True))
        assert_array_equal(np.array([1.0]), dist.shift(array, 1, True))
        assert_array_equal(np.array([1.0]), dist.shift(array, -2, True))
        assert_array_equal(np.array([1.0]), dist.shift(array, -2, True))

        array = np.array([1.0, 2.0, 3.0, 4.0])
        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, 0, True))

        assert_array_equal(np.array([4.0, 1.0, 2.0, 3.0]), 
                dist.shift(array, 1, True))
        assert_array_equal(np.array([2.0, 3.0, 4.0, 1.]), 
                dist.shift(array, -1, True))

        assert_array_equal(np.array([3.0, 4.0, 1.0, 2.0]), 
                dist.shift(array, 2, True))
        assert_array_equal(np.array([3.0, 4.0, 1.0, 2.0]), 
                dist.shift(array, -2, True))

        assert_array_equal(np.array([2.0, 3.0, 4.0, 1.0]), 
                dist.shift(array, 3, True))
        assert_array_equal(np.array([4.0, 1.0, 2.0, 3.0]), 
                dist.shift(array, -3, True))
        
        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, 4, True))
        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, -4, True))
        
        assert_array_equal(np.array([4.0, 1.0, 2.0, 3.0]), 
                dist.shift(array, 5, True))
        assert_array_equal(np.array([2.0, 3.0, 4.0, 1.]), 
                dist.shift(array, -5, True))

        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, 8, True))
        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, -8, True))
    
    def test_shift_drop(self):
        array = np.array([1.0])
        assert_array_equal(np.array([1.0]), dist.shift(array, 0, False))
        assert_array_equal(np.array([0.0]), dist.shift(array, 1, False))
        assert_array_equal(np.array([0.0]), dist.shift(array, 1, False))
        assert_array_equal(np.array([0.0]), dist.shift(array, -2, False))
        assert_array_equal(np.array([0.0]), dist.shift(array, -2, False))

        array = np.array([1.0, 2.0, 3.0, 4.0])
        assert_array_equal(np.array([1.0, 2.0, 3.0, 4.0]), 
                dist.shift(array, 0, False))

        assert_array_equal(np.array([0.0, 1.0, 2.0, 3.0]), 
                dist.shift(array, 1, False))
        assert_array_equal(np.array([2.0, 3.0, 4.0, 0.0]), 
                dist.shift(array, -1, False))

        assert_array_equal(np.array([0.0, 0.0, 1.0, 2.0]), 
                dist.shift(array, 2, False))
        assert_array_equal(np.array([3.0, 4.0, 0.0, 0.0]), 
                dist.shift(array, -2, False))

        assert_array_equal(np.array([0.0, 0.0, 0.0, 1.0]), 
                dist.shift(array, 3, False))
        assert_array_equal(np.array([4.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, -3, False))
        
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, 4, False))
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, -4, False))
        
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, 5, False))
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, -5, False))
        
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, 50, False))
        assert_array_equal(np.array([0.0, 0.0, 0.0, 0.0]), 
                dist.shift(array, -50, False))

    #def test_shift_all(self):
    #    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #    s = np.array([1, 2])
    #
    #    expected = np.array([[3.0, 1.0, 2.0], [5.0, 6.0, 4.0]])
    #    assert_array_almost_equal(expected, dist.shift_all(m, s, True)[0])

    def test_inner_prod(self):
        array1 = np.array([])
        array2 = np.array([])
        self.assertEqual(0, dist.inner_prod(array1, array2))
                
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([2.0, 3.0, 4.0])
        self.assertEqual(sum(array1 * array2), dist.inner_prod(array1, array2))
        
        self.assertEqual(sum(array1 ** 2), dist.inner_prod(array1, array1))

    def test_sqsum(self):
        array = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(array ** 2), dist.sqsum(array), 4)
        
        array = np.array([2.0])
        self.assertEqual(4, dist.sqsum(array))

        array = np.array([])
        self.assertEqual(0, dist.sqsum(array))
    
    def test_shift_dist(self):
        array1 = np.array([])
        array2 = np.array([])
        self.assertEqual(0, dist.shift_dist(array1, array2, 0))
        
        array1 = np.array([0., 0.])
        array2 = np.array([0., 0.])
        self.assertEqual(0, dist.shift_dist(array1, array2, 0))
        
        array1 = np.array([1., 2.])
        array2 = np.array([0., 0.])
        self.assertEqual(1, dist.shift_dist(array1, array2, 0))
        
        array1 = np.array([0., 0.])
        array2 = np.array([1., 2.])
        self.assertEqual(1, dist.shift_dist(array1, array2, 0))

        array1 = np.array([2.0, 3.0, 4.0])
        array2 = np.array([3.0, 4.0, 0.0])

        self.assertEqual(0, dist.shift_dist(array1, array1, 0))
        self.assertEqual(0, dist.shift_dist(array2, array2, 0))
        
        expected = 2 / sqrt(29)
        self.assertAlmostEqual(expected, dist.shift_dist(array1, array2, 1, False))
        
        expected = 2 / sqrt(29)
        self.assertAlmostEqual(expected, dist.shift_dist(array1, array2, 1, True))
    
    def test_dist(self):
        array1 = np.array([])
        array2 = np.array([])
        self.assertEqual(0, dist.dist(array1, array2))
        
        array1 = np.array([0., 0.])
        array2 = np.array([0., 0.])
        self.assertEqual(0, dist.dist(array1, array2))
        
        array1 = np.array([1., 2.])
        array2 = np.array([0., 0.])
        self.assertEqual(1, dist.dist(array1, array2))
       
        array1 = np.array([0., 0.])
        array2 = np.array([1., 2.])
        self.assertEqual(1, dist.dist(array1, array2))
        
        array1 = np.array([2.0, 3.0, 4.0])
        array2 = np.array([3.0, 4.0, 0.0])

        self.assertEqual(0, dist.dist(array1, array1))
        self.assertEqual(0, dist.dist(array2, array2))
        
        expected = 2 / sqrt(29)
        self.assertAlmostEqual(expected, dist.dist(array1, array2, True))
        self.assertAlmostEqual(expected, dist.dist(array2, array1, True))

    def test_dist_all(self):
        m1 = np.array([[0.0], [0.0]])
        m2 = np.array([[0.0], [0.0]])

        expected = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert_array_equal(expected, dist.dist_all(m1, m2)[0])
        assert_array_equal(expected, dist.dist_all(m1, m2)[1])

        m1 = np.array([[1.0], [1.0]])
        m2 = np.array([[0.0], [0.0]])
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert_array_equal(expected, dist.dist_all(m1, m2)[0])

        m1 = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 0.0]])
        m2 = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 0.0]])
        expected = np.array([[0.0, 2/sqrt(29)], [2/sqrt(29), 0.0]])
        assert_array_almost_equal(expected, dist.dist_all(m1, m2, True)[0])

if __name__ == "__main__":
    unittest.main()
