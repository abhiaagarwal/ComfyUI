from typing import Any, Dict, List, Tuple, Union
from typing_extensions import TypedDict

from comfy.types import InputParams

class NodeInfo(TypedDict):
    """Internal representation of a node."""
    input: InputParams
    input_order: Dict[str, List[str]]
    output: Union[Tuple[str, ...], Tuple[()]]
    output_is_list: Tuple[bool, ...]
    output_name: Union[Tuple[str, ...], Tuple[()]]
    name: str
    display_name: str
    description: str
    category: str
    output_node: bool