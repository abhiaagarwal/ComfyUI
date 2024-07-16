from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import TypeGuard

from comfy.types import ComfyNodeV1, Prompt, PromptInput

# this is a big of a hack — while it's actually a list, to make type inference work, we "claim" it's a tuple.
def is_link(obj: Union[Any, List[Any]]) -> TypeGuard[Tuple[str, int]]:
    if not isinstance(obj, list):
        return False
    if len(obj) != 2:
        return False
    if not isinstance(obj[0], str):
        return False
    if not isinstance(obj[1], int) and not isinstance(obj[1], float):
        return False
    return True

# The GraphBuilder is just a utility class that outputs graphs in the form expected by the ComfyUI back-end
class GraphBuilder:
    _default_prefix_root = ""
    _default_prefix_call_index = 0
    _default_prefix_graph_index = 0

    def __init__(self, prefix: Optional[str] = None):
        if prefix is None:
            self.prefix = GraphBuilder.alloc_prefix()
        else:
            self.prefix = prefix
        self.nodes: Dict[str, Node] = {}
        self.id_gen = 1

    @classmethod
    def set_default_prefix(cls, prefix_root: str, call_index: int, graph_index: int = 0):
        cls._default_prefix_root = prefix_root
        cls._default_prefix_call_index = call_index
        cls._default_prefix_graph_index = graph_index

    @classmethod
    def alloc_prefix(cls, root: Optional[str] =None, call_index: Optional[int]=None, graph_index: Optional[int]=None) -> str:
        if root is None:
            root = GraphBuilder._default_prefix_root
        if call_index is None:
            call_index = GraphBuilder._default_prefix_call_index
        if graph_index is None:
            graph_index = GraphBuilder._default_prefix_graph_index
        result = f"{root}.{call_index}.{graph_index}."
        GraphBuilder._default_prefix_graph_index += 1
        return result

    def node(self, class_type: type[ComfyNodeV1], id: Optional[str]=None, **kwargs: Any) -> Node:
        if id is None:
            id = str(self.id_gen)
            self.id_gen += 1
        id = self.prefix + id
        if id in self.nodes:
            return self.nodes[id]

        node = Node(id, class_type, kwargs)
        self.nodes[id] = node
        return node

    def lookup_node(self, id: str) -> Optional[Node]:
        id = self.prefix + id
        return self.nodes.get(id, None)

    def finalize(self) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for node_id, node in self.nodes.items():
            output[node_id] = node.serialize()
        return output

    def replace_node_output(self, node_id: str, index: int, new_value: Optional[int]) -> None:
        node_id = self.prefix + node_id
        to_remove: List[Tuple[Node, str]] = []
        for node in self.nodes.values():
            for key, value in node.inputs.items():
                if is_link(value) and value[0] == node_id and value[1] == index:
                    if new_value is None:
                        to_remove.append((node, key))
                    else:
                        node.inputs[key] = new_value
        for node, key in to_remove:
            del node.inputs[key]

    def remove_node(self, id: str):
        id = self.prefix + id
        del self.nodes[id]

class Node:
    def __init__(self, id: str, class_type: type[ComfyNodeV1], inputs: Dict[str, Any]):
        self.id = id
        self.class_type = class_type
        self.inputs = inputs
        self.override_display_id: Optional[str] = None

    def out(self, index: str) -> Tuple[str, str]:
        return (self.id, index)

    def set_input(self, key: str, value: Any) -> None:
        if value is None:
            if key in self.inputs:
                del self.inputs[key]
        else:
            self.inputs[key] = value

    def get_input(self, key: str) -> None:
        return self.inputs.get(key)

    def set_override_display_id(self, override_display_id: str):
        self.override_display_id = override_display_id

    def serialize(self) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {
            "class_type": self.class_type,
            "inputs": self.inputs
        }
        if self.override_display_id is not None:
            serialized["override_display_id"] = self.override_display_id
        return serialized

def add_graph_prefix(graph: Prompt, outputs, prefix: str):
    # Change the node IDs and any internal links
    new_graph = {}
    for node_id, node_info in graph.items():
        # Make sure the added nodes have unique IDs
        new_node_id = prefix + node_id
        new_node: PromptInput = { "class_type": node_info["class_type"], "inputs": {} }
        for input_name, input_value in node_info.get("inputs", {}).items():
            if is_link(input_value):
                new_node["inputs"][input_name] = [prefix + input_value[0], input_value[1]]
            else:
                new_node["inputs"][input_name] = input_value
        new_graph[new_node_id] = new_node

    # Change the node IDs in the outputs
    new_outputs = []
    for n in range(len(outputs)):
        output = outputs[n]
        if is_link(output):
            new_outputs.append([prefix + output[0], output[1]])
        else:
            new_outputs.append(output)

    return new_graph, tuple(new_outputs)
