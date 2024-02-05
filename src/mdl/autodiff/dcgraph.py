from __future__ import annotations
from mdl.tensor import Tensor
from typing import List, Optional

class DCGraph:
    
    _instance: Optional[DCGraph] = None
    
    # Singleton pattern (ensuring that only one instance of the DCGraph)
    # is created. If an instance exists, return that instance instead.
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DCGraph, cls).__new__(cls)
            cls._instance.tensor_nodes: List[Tensor] = []
            cls._instance.ordered_nodes: List[Tensor] = []
        return cls._instance
    
    @property
    def tensor_nodes(self) -> List[Tensor]:
        return self.tensor_nodes
    
    @property
    def ordered_nodes(self) -> List[Tensor]:
        return self.ordered_nodes
    
    @ordered_nodes.setter
    def ordered_nodes(self, tensor_nodes: List[Tensor]):
        self.ordered_nodes = tensor_nodes

    def add_tensor_node(self, tensor: Tensor) -> None:
            self.tensor_nodes.append(tensor)

    def topological_sort(self) -> None:
        visited = set()
        stack = []

        def visit(node, parents):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    visit(child, parents + [node])
                stack.append((node, parents))

        for node in self.tensor_nodes:
            visit(node, [])

        stack.sort(key=lambda x: len(x[1]))  # Sort by the number of parents

        self.ordered_nodes =  [tensor_node for tensor_node, _ in stack]