import torch
from typing import Callable, Optional, List, Tuple, Union, Dict, Any
from typing_extensions import TypeAlias, NotRequired, TypedDict, Protocol


class UnetApplyFunction(Protocol):
    """Function signature protocol on comfy.model_base.BaseModel.apply_model"""

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...


class UnetApplyConds(TypedDict):
    """Optional conditions for unet apply function."""

    c_concat: Optional[torch.Tensor]
    c_crossattn: Optional[torch.Tensor]
    control: Optional[torch.Tensor]
    transformer_options: Optional[dict]


class UnetParams(TypedDict):
    # Tensor of shape [B, C, H, W]
    input: torch.Tensor
    # Tensor of shape [B]
    timestep: torch.Tensor
    c: UnetApplyConds
    # List of [0, 1], [0], [1], ...
    # 0 means conditional, 1 means conditional unconditional
    cond_or_uncond: List[int]


UnetWrapperFunction = Callable[[UnetApplyFunction, UnetParams], torch.Tensor]

InputParamDefinition: TypeAlias = Dict[
    str, Union[Tuple[str | List[str]], Tuple[str | List[str], Dict[str, Any]]]
]


class InputParams(TypedDict):
    required: InputParamDefinition
    optional: NotRequired[InputParamDefinition]
    hidden: NotRequired[InputParamDefinition]


class ComfyNodeV1(Protocol):
    RETURN_TYPES: Union[Tuple[str, ...], Tuple[()]]
    RETURN_NAMES: NotRequired[Tuple[str, ...]]
    FUNCTION: str
    CATEGORY: NotRequired[str]
    OUTPUT_IS_LIST: NotRequired[Tuple[bool, ...]]
    DESCRIPTION: NotRequired[str]
    OUTPUT_NODE: NotRequired[bool] = False
    INPUT_IS_LIST: NotRequired[bool] = False

    @classmethod
    def INPUT_TYPES(
        cls,
    ) -> InputParams: ...

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        inputs: Dict[str, Any],
    ) -> None: ...


class PromptInput(TypedDict):
    class_type: str
    inputs: Dict[str, Any]


Prompt = Dict[str, PromptInput]
