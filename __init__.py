from .hidreamsampler import HiDreamSampler

NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
