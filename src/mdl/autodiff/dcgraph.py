from __future__ import annotations

from typing import List
from typing import Set
from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from mdl.tensor import Tensor


class DCGraph:

    _instance: DCGraph | None = None

    # Singleton pattern (ensuring that only one instance of the DCGraph)
    # is created. If an instance exists, return that instance instead.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tensor_nodes: Set[Tensor, None] = set()
        return cls._instance

    def __str__(self):
        return f"Graph({self.tensor_nodes})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def tensor_nodes(self) -> Set[Tensor]:
        return self.tensor_nodes

    def __len__(self):
        return len(self.tensor_nodes)

    def add_tensor_node(self, tensor: Tensor) -> None:
        self.tensor_nodes.add(tensor)

    def remove_tensor_node(self, tensor: Tensor) -> Tensor:
        self.tensor_nodes.remove(tensor)

    def reset_graph(self):
        self.tensor_nodes = set()

    def topological_sort(self) -> List[Union[Tensor, None]]:
        visited = set()
        stack = []

        def visit(tensor_node, parents):
            if tensor_node not in visited:
                visited.add(tensor_node)
                for child in tensor_node.children:
                    visit(child, parents + [node])
                stack.append((node, parents))

        for node in self.tensor_nodes:
            visit(node, [])

        stack.sort(key=lambda x: len(x[1]))  # Sort by the number of parents

        return [tensor_node for tensor_node, _ in stack]
