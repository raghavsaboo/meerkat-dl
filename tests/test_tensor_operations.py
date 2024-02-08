import unittest
import numpy as np
from mdl.tensor import Tensor

class TestTensorOperations(unittest.TestCase):
    def test_addition(self):
        tensor_a = Tensor(np.array([1, 2, 3]))
        tensor_b = Tensor(np.array([4, 5, 6]))
        result = tensor_a + tensor_b
        expected_result = Tensor(np.array([5, 7, 9]))
        self.assertTrue(np.array_equal(result.data, expected_result.data))

    def test_subtraction(self):
        tensor_a = Tensor(np.array([4, 5, 6]))
        tensor_b = Tensor(np.array([1, 2, 3]))
        result = tensor_a - tensor_b
        expected_result = Tensor(np.array([3, 3, 3]))
        self.assertTrue(np.array_equal(result.data, expected_result.data))

    def test_multiplication(self):
        tensor_a = Tensor(np.array([2, 3, 4]))
        tensor_b = Tensor(np.array([1, 2, 3]))
        result = tensor_a * tensor_b
        expected_result = Tensor(np.array([2, 6, 12]))
        self.assertTrue(np.array_equal(result.data, expected_result.data))

    # Add tests for other operations (Divide, Exp, Log, Sum, Concatenate) here

if __name__ == '__main__':
    unittest.main()