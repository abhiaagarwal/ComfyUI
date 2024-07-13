"""
A high-level overview of what a "node" looks like.

This defines a JSON-schema based representation of a node, which is cached.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from enum import StrEnum
from typing_extensions import Annotated, get_type_hints
from pydantic import BaseModel, Field

from comfy.api.typed_node import ComfyInput, ComfyNode, OldNode, InputType
from comfy.types import ComfyType


class NodeType(StrEnum):
    BUILTIN = "builtin"
    EXTRAS = "extras"
    CUSTOM = "custom"

class Input(BaseModel):
    name: str
    description: Optional[str]
    input_type: InputType
    type_: ComfyType
    choices: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

class Output(BaseModel):
    name: str
    description: Optional[str]
    type_: ComfyType

class NodeRepresentation(BaseModel):
    """A high-level overview of what a "node" looks like. This is used by the execution code."""

    display_name: str
    description: Optional[str]
    version: str
    category: str
    node_type: NodeType

    parameters: List[Input]
    outputs: List[Output]

    """The function that actually executes the relevant code.
    
    The v2 format explicitly disallows the use of `self`, so this is static. For v1 nodes,
    `self` is captured as a partial function to hold it as part of this field.
    """
    function: Callable[[Any], Optional[ComfyType]] = Field(exclude=True)


class NodeRegistry:
    nodes: Dict[str, NodeRepresentation] = {}
    _lock: bool = False

    def register(self, name: str, node: NodeRepresentation) -> None:
        if self._lock:
            raise RuntimeError("Cannot register nodes after the registry is locked.")
        self.nodes[name] = node
    
    def get(self, name: str) -> Optional[NodeRepresentation]:
        if not self._lock:
            raise RuntimeError("Cannot get nodes before the registry is locked.")
        return self.nodes.get(name, None)
    
    def lock(self) -> None:
        self._lock = True
    
    def register_old_node(self, name: str, display_name: str, node: OldNode, node_type: NodeType) -> None:
        """Register a node from the old style."""
        if self._lock:
            raise RuntimeError("Cannot register nodes after the registry is locked.")
        
        inputs = node.INPUT_TYPES()
        required_inputs = inputs["required"]
        required_inputs

    def register_v2_node(self, node: ComfyNode, node_type: NodeType) -> None:
        """Register a node from the new style."""
        f = node.execute
        hints = get_type_hints(f, include_extras=True)
        return_types = hints.pop("return")
        inputs: List[Input] = []
        if len(hints) == 0:
            pass
        else:
            for param, type_val in hints.items():
                if not isinstance(type_val, Annotated):
                    type_str = type_val.__name__
                    inputs.append(
                        Input(
                            name=param,
                            description=None,
                            input_type=InputType.REQUIRED,
                            type_=type_str
                        )
                    )
                else:
                    type_str: str = type_val.__origin__.__name__
                    for annotation in type_val.__metadata__:
                        if isinstance(annotation, ComfyInput):
                            inputs.append(Input(
                                name=param,
                                description=annotation.description,
                                input_type=annotation.input_type,
                                type_=type_str,
                                choices=annotation.choices,
                                extra=annotation.additional_values
                            ))
                            break

@dataclass
class NodeInfo:
    """NodeInfo, as consumed by the frontend.
    
    This should probably be deprecated in the future in favor of the JSON representation of `NodeRepresentation`.
    """
    input: Dict[str, Dict[str, Any]]
    output: Tuple[str, ...]
    output_is_list: Tuple[bool, ...]
    node_class: str
    display_name: str
    description: str
    python_module: str
    category: str
    output_node: bool

    @staticmethod
    def from_node_representation(node: NodeRepresentation) -> NodeInfo:
        return NodeInfo(
            input={i.name: i.dict() for i in node.parameters},
            output=tuple(o.name for o in node.outputs),
            output_is_list=tuple(False for _ in node.outputs),
            node_class=node.display_name,
            display_name=node.display_name,
            description=node.description,
            python_module="",
            category=node.category,
            output_node=False
        )