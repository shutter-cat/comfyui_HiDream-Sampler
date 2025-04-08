import torch
import numpy as np
from PIL import Image
#import comfy.model_management as model_management

import torch
from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

#quantization, not as good as full, but uses way less memory
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=True,
        low_cpu_mem_usage=True,
        quantization_config=TransformersBitsAndBytesConfig(
            load_in_4bit=True,
        ),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda")
    
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer",
        quantization_config=DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
        ), 
        torch_dtype=torch.bfloat16).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer
    
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

def pil2tensor(image: Image.Image):
    """Converts a PIL Image to a ComfyUI-compatible tensor."""
    if image is None:
        return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["full", "dev", "fast"], {"default": "fast"}),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of an astronaut riding a horse on the moon"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffff}),
                # Steps and CFG are now derived from model_type, but can be overridden optionally
                 "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}), # -1 uses config default
                 "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}), # -1 uses config default
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg):
        
        # Load or retrieve cached model
        if model_type not in self._model_cache:
            print(f"Loading model for {model_type}...")
            pipe, config = load_models(model_type)
            self._model_cache[model_type] = (pipe, config)
            print(f"Model for {model_type} cached successfully!")
        else:
            print(f"Using cached model for {model_type}")
            pipe, config = self._model_cache[model_type]
        
        # --- Load Models ---
        # This function now handles caching and device placement
        config = MODEL_CONFIGS[model_type]

        # --- Parse Inputs ---
        width, height = parse_resolution(resolution) 

        num_inference_steps = override_steps if override_steps > 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed ---
        generator = torch.Generator("cuda").manual_seed(seed)
        
        output_images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
        ).images

        print("[HiDream Node] Generation Complete.")

        # --- Convert to ComfyUI Tensor ---
        output_tensor = pil2tensor(output_images[0])

        return (output_tensor,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler"
}