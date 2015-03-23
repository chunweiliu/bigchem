import unittest

import numpy as np

from gpu_tanimoto_popcount import GPUtanimoto


class TestGPUTanimoto(unittest.TestCase):

    def test_zeros(self):
        query = np.array([[0]], np.uint64)
        target = np.array([[0]], np.uint64)
        output = GPUtanimoto(query, target)
        expect = [(0, 1.0)]
        self.assertEqual(output, expect)

    def test_ones(self):
        query = np.array([[1]], np.uint64)
        target = np.array([[1]], np.uint64)
        output = GPUtanimoto(query, target)
        expect = [(0, 1.0)]
        self.assertEqual(output, expect)

    def test_double(self):
        query = np.array([[0, 1]], np.uint64)
        target = np.array([[0, 1], [1, 0]], np.uint64)
        output = GPUtanimoto(query, target)
        expect = [(0, 1.0), (1, 0.0)]
        self.assertEqual(output, expect)

if __name__ == '__main__':
    unittest.main()
