from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import (
    Annotated,
    Protocol,
    TypedDict,
    TypeAlias,
    ClassVar,
)

from comfy.types import ComfyType
import nodes

class InputType(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    HIDDEN = "hidden"    


@dataclass
class ComfyInput:
    """Metadata used by the UI to handle the input."""

    """An optional description of the input, ie the purpose it serves"""
    description: Optional[str] = None

    input_type: InputType = InputType.REQUIRED

    choices: Optional[List[str]] = None

    """An optional default value for the input"""
    additional_values: Optional[Dict[str, Any]] = None

@dataclass
class ComfyOutput:
    """An optional description of the output, ie the purpose it serves"""

    description: Optional[str] = None


@dataclass
class NodeMetadata:
    """Additional metadata for the node."""
    author: str

    """A semver-compliant version string for the node. Can be used to make breaking changes to the node."""
    version: str


class ComfyNode(Protocol):
    name: Optional[str] = None
    display_name: Optional[str] = None
    category: str
    description: str

    @classmethod
    def execute(
        cls, *args: Union[ComfyType, Annotated[ComfyType, ComfyInput]]
    ) -> Union[
        None,
        Any,
        Annotated[ComfyType, ComfyOutput],
        Tuple[Union[ComfyType, Annotated[ComfyType, ComfyOutput]], ...],
    ]:
        ...


ParamDefinition: TypeAlias = Dict[str, Union[Tuple[str], List[str], Tuple[str, Dict[str, Any]]]]

class ParamTypes(TypedDict):
    required: ParamDefinition
    optional: ParamDefinition
    hidden: ParamDefinition

class OldNode(Protocol):
    RETURN_TYPES: ClassVar[Union[Tuple[str], Tuple[str, ...], Tuple[()]]]
    FUNCTION: str
    CATEGORY: str
    OUTPUT_IS_LIST: Tuple[bool, ...]
    DESCRIPTION: str
    RELATIVE_PYTHON_MODULE: str
    OUTPUT_NODE: bool
    INPUT_IS_LIST: bool

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> ParamTypes:
        ...


def node_info(node_class: str) -> NodeInfo:
    obj_class: OldNode = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = NodeInfo(
        input=obj_class.INPUT_TYPES(),
        output=obj_class.RETURN_TYPES,
        output_is_list=obj_class.OUTPUT_IS_LIST
        if hasattr(obj_class, "OUTPUT_IS_LIST")
        else tuple([False] * len(obj_class.RETURN_TYPES)),
        node_class=node_class,
        display_name=nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class]
        if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys()
        else node_class,
        description=obj_class.DESCRIPTION if hasattr(obj_class, "DESCRIPTION") else "",
        python_module=getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes"),
        category=obj_class.CATEGORY if hasattr(obj_class, "CATEGORY") else "sd",
        output_node=True
        if hasattr(obj_class, "OUTPUT_NODE") and obj_class.OUTPUT_NODE
        else False,
    )
    return info
