from .hidreamsampler import HiDreamSampler, HiDreamSamplerAdvanced

NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler,
    "HiDreamSamplerAdvanced": HiDreamSamplerAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler",
    "HiDreamSamplerAdvanced": "HiDream Sampler (Advanced)"
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
