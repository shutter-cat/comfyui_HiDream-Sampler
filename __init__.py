from .hidreamsampler import HiDreamSampler

NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler"
}

ascii_art = """
HiDreamSampler is brought to you by

       _  _                                                                     
      (▒)(▒)                                _  _  _                                 
         (▒)   _      _    _  _   _  _    _(▒)(▒)(▒)_    _  _    _  _  _      
         (▒)  (▒)    (▒)  (▒)(▒)_(▒)(▒)  (▒)    _  (▒) _(▒)(▒)_ (▒)(▒)(▒)_    
         (▒)  (▒)    (▒)  (▒)   (▒)   (▒)      (▒)(▒) (▒)    (▒)(▒)     (▒)      
       _ (▒) _(▒)_  _(▒)_ (▒)   (▒)   (▒)(▒)_  _  _(▒)(▒)_  _(▒)(▒)     (▒)   
      (▒)(▒)(▒) (▒)(▒) (▒)(▒)   (▒)   (▒)  (▒)(▒)(▒)    (▒)(▒)  (▒)     (▒)                                                                                              
"""
print(f"\033[92m{ascii_art}\033[0m")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
