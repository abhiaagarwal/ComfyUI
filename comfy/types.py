from __future__ import annotations

import torch
from typing import Any, Callable, Optional, List
from typing_extensions import NewType, Protocol, TypedDict
from abc import ABC

from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE as SDVAE, CLIP as SDCLIP

class ComfyType(ABC):
    pass

FLOAT = NewType("FLOAT", float)
ComfyType.register(FLOAT)

MODEL = NewType("MODEL", ModelPatcher)
ComfyType.register(MODEL)

VAE = NewType("VAE", SDVAE)
ComfyType.register(VAE)

CLIP = NewType("CLIP", SDCLIP)
ComfyType.register(CLIP)

STRING = NewType("STRING", str)
ComfyType.register(STRING)

CONDITIONING = NewType("CONDITIONING", torch.Tensor)
ComfyType.register(CONDITIONING)

IMAGE = NewType("IMAGE", torch.Tensor)
ComfyType.register(IMAGE)


class UnetApplyFunction(Protocol):
    """Function signature protocol on comfy.model_base.BaseModel.apply_model"""

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        ...


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
