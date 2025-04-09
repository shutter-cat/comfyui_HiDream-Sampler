import torch
import numpy as np
from PIL import Image
# import comfy.model_management as mm # Not using manual management here
import comfy.utils # Keep for progress bar potentially

# Diffusers/Transformers imports
from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Quantization configs
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
HIDREAM_FP8_MODEL_PATH = "shuttleai/HiDream-I1-Full-FP8" # Specific path for FP8 version

# Model configurations - Added 'full-fp8'
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_fp8": False
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler,
        "is_fp8": False
    },
    "full-fp8": { # New entry for the FP8 model
        "path": HIDREAM_FP8_MODEL_PATH,
        "guidance_scale": 5.0, # Assuming same params as 'full'
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler, # Assuming same scheduler
        "is_fp8": True # Mark this as FP8
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_fp8": False
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

# Define BitsAndBytes configs
bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True) # For non-FP8 transformers
model_dtype = torch.bfloat16 # Keep bfloat16

# Load models - Reverted to original structure + FP8 condition
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    is_fp8_transformer = config.get("is_fp8", False)

    # Select the correct scheduler class from config
    scheduler_class = config["scheduler_class"]
    scheduler = scheduler_class(
        num_train_timesteps=1000, # Or get from model config if available
        shift=config["shift"],
        use_dynamic_shifting=False # Or get from config
    )

    print(f"[HiDream Node] Loading Tokenizer: {LLAMA_MODEL_NAME}")
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)

    print(f"[HiDream Node] Loading 4-bit LLM Text Encoder: {LLAMA_MODEL_NAME}")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        # output_attentions=True, # Keep disabled unless needed
        low_cpu_mem_usage=True,
        quantization_config=bnb_llm_config, # Use 4-bit for LLM
        torch_dtype=model_dtype,
        attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
        ).to("cuda") # Original placement

    # --- Conditional Transformer Loading ---
    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True
    }
    if is_fp8_transformer:
        print(f"[HiDream Node] Loading FP8 Diffusion Transformer: {pretrained_model_name_or_path}")
        # NO quantization_config for FP8
    else:
        print(f"[HiDream Node] Loading 4-bit Diffusion Transformer: {pretrained_model_name_or_path}")
        transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config # Apply 4-bit config

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        **transformer_load_kwargs
    ).to("cuda") # Original placement

    # --- Pipeline Loading (Original Structure) ---
    # Pass the correct model path here too
    print(f"[HiDream Node] Loading Pipeline: {pretrained_model_name_or_path}")
    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=model_dtype
        # Note: If the FP8 model needs specific variant flags (like variant='fp8'),
        # they might need to be passed here too if the pipeline doesn't infer it.
        # Let's try without first.
    ).to("cuda", model_dtype) # Original placement

    # Overwrite the pipeline's transformer with the one we explicitly loaded
    # (ensures correct quantization/FP8 is used)
    print("[HiDream Node] Assigning explicitly loaded transformer to pipeline.")
    pipe.transformer = transformer

    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    # Reverted to simpler original logic, assuming W x H in string -> H, W for function return
    if "1024 × 1024" in resolution_str: return 1024, 1024
    elif "768 × 1360" in resolution_str: return 1360, 768 # H, W
    elif "1360 × 768" in resolution_str: return 768, 1360 # H, W
    elif "880 × 1168" in resolution_str: return 1168, 880 # H, W
    elif "1168 × 880" in resolution_str: return 880, 1168 # H, W
    elif "1248 × 832" in resolution_str: return 832, 1248 # H, W
    elif "832 × 1248" in resolution_str: return 1248, 832 # H, W
    else: return 1024, 1024

def pil2tensor(image: Image.Image):
    if image is None: return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition (Original Structure) ---
class HiDreamSampler:

    _model_cache = {} # Keep original caching mechanism

    @classmethod
    def INPUT_TYPES(s):
        model_type_options = list(MODEL_CONFIGS.keys())
        default_model = "fast" if "fast" in model_type_options else model_type_options[0]
        return {
            "required": {
                "model_type": (model_type_options, {"default": default_model}), # Includes 'full-fp8'
                "prompt": ("STRING", {"multiline": True, "default": "A photo of an astronaut riding a horse on the moon"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use wider seed range
                 "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                 "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg):

        # Load or retrieve cached model (Original Caching Logic)
        if model_type not in self._model_cache:
            print(f"Loading model for {model_type}...")
            try:
                pipe, config = load_models(model_type)
                self._model_cache[model_type] = (pipe, config)
                print(f"Model for {model_type} cached successfully!")
            except Exception as e:
                print(f"!!! ERROR loading model {model_type}: {e}")
                # Clear cache entry if loading failed
                if model_type in self._model_cache: del self._model_cache[model_type]
                import traceback
                traceback.print_exc()
                # Re-raise or return error state? Re-raising for now.
                raise e # Propagate error to ComfyUI
        else:
            print(f"Using cached model for {model_type}")
            pipe, config = self._model_cache[model_type] # Get from cache

        # --- Parse Inputs ---
        # Get config *again* from dict in case it wasn't cached (or use cached one)
        # This seems slightly redundant with the caching but matches original structure closer
        config_from_dict = MODEL_CONFIGS[model_type]
        height, width = parse_resolution(resolution) # Returns H, W

        # Use config from cache/load for steps/cfg defaults
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed ---
        # Ensure generator is on CUDA if pipe is on CUDA
        device = pipe.device # Get device from the loaded pipeline
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"[HiDream Node] Starting generation: H={height}, W={width}, Steps={num_inference_steps}, CFG={guidance_scale}, Seed={seed}")
        # Add progress bar using comfy.utils
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        def progress_callback(step, timestep, latents):
            pbar.update(1)

        # --- Run Inference ---
        # Use inference_mode or no_grad for potentially lower memory during inference itself
        with torch.inference_mode():
             output_images = pipe(
                 prompt=prompt, # Using prompt, not embeds, as per original structure
                 height=height,
                 width=width,
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps,
                 num_images_per_prompt=1,
                 generator=generator,
                 callback_steps = 1, # Call callback every step
                 callback = progress_callback,
             ).images

        pbar.update_absolute(num_inference_steps) # Ensure bar completes
        print("[HiDream Node] Generation Complete.")

        # --- Convert to ComfyUI Tensor ---
        if not output_images:
             print("[HiDream Node] ERROR: No images were generated.")
             # Return blank image to avoid breaking workflow
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)

        output_tensor = pil2tensor(output_images[0])

        # --- Model Cleanup (Implicit) ---
        # Relying on ComfyUI's management and Python's garbage collection
        # as we are not doing manual load/unload within generate anymore.
        # The cache keeps the model loaded.

        return (output_tensor,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler (FP8 Option)" # Updated name
}
