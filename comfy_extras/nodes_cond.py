from typing_extensions import Annotated
from comfy.api.typed_node import (
    ComfyNode,
    ComfyInput,
    CLIP,
    CONDITIONING,
    STRING,
)


# class CLIPTextEncodeControlnet:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {"clip": ("CLIP", ), "conditioning": ("CONDITIONING", ), "text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
#     RETURN_TYPES = ("CONDITIONING",)
#     FUNCTION = "encode"

#     CATEGORY = "_for_testing/conditioning"

#     def encode(self, clip, conditioning, text):
#         tokens = clip.tokenize(text)
#         cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
#         c = []
#         for t in conditioning:
#             n = [t[0], t[1].copy()]
#             n[1]['cross_attn_controlnet'] = cond
#             n[1]['pooled_output_controlnet'] = pooled
#             c.append(n)
#         return (c, )

# NODE_CLASS_MAPPINGS = {
#     "CLIPTextEncodeControlnet": CLIPTextEncodeControlnet
# }


class CLIPTextEncodeControlnet(ComfyNode):
    @property
    def category(self) -> str:
        return "_for_testing/conditioning"

    def execute(
        self,
        clip: CLIP,
        conditioning: CONDITIONING,
        text: Annotated[
            STRING,
            ComfyInput(
                additional_values={"multiline": True, "dynamicPrompts": True},
            ),
        ],
    ) -> CONDITIONING:
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["cross_attn_controlnet"] = cond
            n[1]["pooled_output_controlnet"] = pooled
            c.append(n)
        return c
