import numpy as np
import pytest
from mdl.tensor import Tensor

def test_tensor_creation():
    data = np.array([1, 2, 3])
    tensor = Tensor(data)
    assert np.array_equal(tensor.data, data)
    assert tensor.requires_grad is False
    assert tensor.should_broadcast is True

def test_tensor_operations():
    tensor_a = Tensor(np.array([1, 2, 3]))
    tensor_b = Tensor(np.array([4, 5, 6]))

    # Addition
    result_add = tensor_a + tensor_b
    expected_result_add = Tensor(np.array([5, 7, 9]))
    assert np.array_equal(result_add.data, expected_result_add.data)

    # Subtraction
    result_sub = tensor_a - tensor_b
    expected_result_sub = Tensor(np.array([-3, -3, -3]))
    assert np.array_equal(result_sub.data, expected_result_sub.data)

    # Multiplication
    result_mul = tensor_a * tensor_b
    expected_result_mul = Tensor(np.array([4, 10, 18]))
    assert np.array_equal(result_mul.data, expected_result_mul.data)

    # Division
    result_div = tensor_a / tensor_b
    expected_result_div = Tensor(np.array([0.25, 0.4, 0.5]))
    assert np.array_equal(result_div.data, expected_result_div.data)

    # Exponentiation
    result_pow = tensor_a ** 2
    expected_result_pow = Tensor(np.array([1, 4, 9]))
    assert np.array_equal(result_pow.data, expected_result_pow.data)

    # Logarithm
    result_log = tensor_a.log()
    expected_result_log = Tensor(np.array([0, 0.6931, 1.0986]))
    assert np.allclose(result_log.data, expected_result_log.data, atol=1e-03)

    # Sum
    result_sum = tensor_a.sum()
    expected_result_sum = Tensor(np.array(6))
    assert np.array_equal(result_sum.data, expected_result_sum.data)

def test_tensor_operations_with_simple_backward_pass():
    tensor_a = Tensor(np.array([1, 2, 3]), requires_grad=True)
    tensor_b = Tensor(np.array([4, 5, 6]), requires_grad=True)
    result_add = tensor_a + tensor_b

    # Simulating backward pass
    output_grad = np.array([1, 1, 1])
    result_add.backward(output_grad)

    # Check gradients
    assert np.array_equal(tensor_a.grad, output_grad)
    assert np.array_equal(tensor_b.grad, output_grad)

def test_complex_backward_pass():
    # Complex scenario with multiple operations and backward pass
    tensor_a = Tensor(np.array([1, 2, 3]), requires_grad=True)
    tensor_b = Tensor(np.array([4, 5, 6]), requires_grad=True)
    tensor_c = Tensor(np.array([7, 8, 9]), requires_grad=True)

    # Operations
    result_add = tensor_a + tensor_b
    result_mul = result_add * tensor_c
    result_sum = result_mul.sum()

    # Simulating backward pass
    output_grad = np.array(1.0)
    result_sum.backward(output_grad)

    # Check gradients
    assert np.array_equal(tensor_a.grad, np.array([7, 8, 9]))
    assert np.array_equal(tensor_b.grad, np.array([7, 8, 9]))
    assert np.array_equal(tensor_c.grad, np.array([5, 7, 9]))
    assert np.array_equal(result_add.grad, np.array([7, 8, 9]))


if __name__ == '__main__':
    pytest.main()