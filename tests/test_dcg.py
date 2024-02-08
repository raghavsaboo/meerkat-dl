import unittest
import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Tensor
from collections import deque

class TestDCGraph(unittest.TestCase):
    def test_add_tensor_node(self):
        dc_graph = DCGraph()
        tensor_a = Tensor(np.array([1, 2, 3]))
        dc_graph.add_tensor_node(tensor_a)
        self.assertIn(tensor_a, dc_graph.tensor_nodes)
        dc_graph.reset_graph()

    def test_remove_tensor_node(self):
        dc_graph = DCGraph()
        tensor_a = Tensor(np.array([1, 2, 3]))
        dc_graph.add_tensor_node(tensor_a)
        dc_graph.remove_tensor_node(tensor_a)
        self.assertNotIn(tensor_a, dc_graph.tensor_nodes)
        dc_graph.reset_graph()

    def test_add_edge(self):
        dc_graph = DCGraph()
        tensor_a = Tensor(np.array([1, 2, 3]))
        tensor_b = Tensor(np.array([4, 5, 6]))
        dc_graph.add_edge(tensor_a, [tensor_b])
        self.assertIn(tensor_a, dc_graph.tensor_nodes)
        self.assertIn(tensor_b, dc_graph.tensor_nodes)
        self.assertIn(tensor_b, tensor_a.parent_tensors)
        self.assertIn(tensor_a, tensor_b.child_tensors)
        dc_graph.reset_graph()

    def test_reset_graph(self):
        dc_graph = DCGraph()
        tensor_a = Tensor(np.array([1, 2, 3]))
        dc_graph.add_tensor_node(tensor_a)
        dc_graph.reset_graph()
        self.assertEqual(len(dc_graph.tensor_nodes), 0)
        dc_graph.reset_graph()

    def test_topological_sort(self):
        dc_graph = DCGraph()
        tensor_a = Tensor(np.array([1, 2, 3]))
        tensor_b = Tensor(np.array([4, 5, 6]))
        tensor_c = Tensor(np.array([7, 8, 9]))
        dc_graph.add_edge(tensor_a, [tensor_b])
        dc_graph.add_edge(tensor_a, [tensor_c])
        tensor_queue = dc_graph.topological_sort(tensor_a)
        self.assertEqual(tensor_queue, deque(reversed([tensor_c, tensor_b, tensor_a])))
        dc_graph.reset_graph()

if __name__ == '__main__':
    unittest.main()