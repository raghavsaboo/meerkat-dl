from __future__ import annotations

from typing import Deque
from typing import List
from typing import Set
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from mdl.tensor import Tensor, Parameter

from collections import deque


class DCGraph:

    _instance: DCGraph | None = None

    # Singleton pattern (ensuring that only one instance of the DCGraph)
    # is created. If an instance exists, return that instance instead.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tensor_nodes: Set[Union[Tensor, None]] = set()
        return cls._instance

    def __str__(self):
        return f"Graph({self.tensor_nodes})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __len__(self):
        return len(self.tensor_nodes)

    @property
    def tensor_nodes(self):
        return self._tensor_nodes

    @tensor_nodes.setter
    def tensor_nodes(self, value: Set[Union[Tensor, None]]):
        self._tensor_nodes = value

    def zero_grad(self):
        for tensor in self.tensor_nodes:
            tensor.zero_grad()

    def add_tensor_node(self, tensor: Tensor) -> None:
        if tensor not in self.tensor_nodes:
            self.tensor_nodes.add(tensor)

    def remove_tensor_node(self, tensor: Tensor) -> None:
        self.tensor_nodes.remove(tensor)

    def add_edge(self, result: Tensor, operands: List[Tensor]) -> None:
        self.add_tensor_node(result)

        for operand in operands:
            self.add_tensor_node(operand)
            operand.add_child_tensor(result)
            result.add_parent_tensor(operand)

    def reset_graph(self):
        self.tensor_nodes.clear()

    def backpropogate(self, tensor: Tensor) -> None:
        tensor_queue = self.topological_sort(tensor)

        while tensor_queue:
            current = tensor_queue.popleft()
            current.backprop_calculation()

    def topological_sort(self, tensor: Tensor) -> Deque[Union[Tensor, Parameter]]:
        visited = set(tensor.child_tensors)
        tensor_queue: Deque[Union[Tensor, Parameter]] = deque()

        def topo_sort(tensor):
            if tensor not in visited:
                for child in tensor.child_tensors:
                    topo_sort(child)
                visited.add(tensor)
                tensor_queue.append(tensor)
                for parent in tensor.parent_tensors:
                    topo_sort(parent)

        topo_sort(tensor)

        return tensor_queue
