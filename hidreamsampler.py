import torch
import numpy as np
from PIL import Image
import comfy.utils
import gc

# Diffusers/Transformers imports
# Ensure these imports match the ones used by the azaneko script if different
from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Quantization configs (only needed for non-NF4/non-FP8)
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
FP8_MODEL_PATH = "shuttleai/HiDream-I1-Full-FP8"

ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4

# --- Model Configurations ---
# Added nf4 variants, updated flags and paths
MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler,
        "is_nf4": True, "is_fp8": False
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_nf4": True, "is_fp8": False
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_nf4": True, "is_fp8": False
    },
     # --- FP8 Model ---
    "full-fp8": {
        "path": FP8_MODEL_PATH,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler,
        "is_nf4": False, "is_fp8": True
    },
    # --- Original/BNB Models ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler,
        "is_nf4": False, "is_fp8": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_nf4": False, "is_fp8": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler,
        "is_nf4": False, "is_fp8": False
    }
}

# Define BitsAndBytes configs (only for non-NF4/FP8)
bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
model_dtype = torch.bfloat16

# --- Loading Function (Handles NF4, FP8, and default BNB) ---
def load_models(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    is_fp8 = config.get("is_fp8", False)
    scheduler_class = config["scheduler_class"]

    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, FP8: {is_fp8}")

    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {
        "output_hidden_states": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": model_dtype,
    }
    if is_nf4:
        llama_model_name = NF4_LLAMA_MODEL_NAME
        print(f"Loading NF4-compatible LLM (GPTQ): {llama_model_name}")
        # NF4 script uses device_map="auto" for GPTQ model
        text_encoder_load_kwargs["device_map"] = "auto"
        # Don't use BNB config for GPTQ
    else:
        llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
        print(f"Loading Standard LLM (4-bit BNB): {llama_model_name}")
        text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
        # We'll move this one manually later

    print(f"Loading Tokenizer: {llama_model_name}")
    # Use the *same* tokenizer name as the text encoder being loaded
    tokenizer = PreTrainedTokenizerFast.from_pretrained(llama_model_name, use_fast=False)

    print(f"Loading Text Encoder: {llama_model_name}")
    text_encoder = LlamaForCausalLM.from_pretrained(
        llama_model_name,
        **text_encoder_load_kwargs
    )

    # Manually move non-NF4 text encoder; device_map handles NF4
    if not is_nf4:
        print("Moving standard LLM to CUDA...")
        text_encoder.to("cuda")

    print("✅ Text encoder loaded!")
    allocated_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(VRAM allocated: {allocated_mb:.2f} MB)")


    # --- 2. Load Transformer (Conditional) ---
    transformer_load_kwargs = {
        "subfolder": "transformer",
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True
    }
    if is_nf4:
        print(f"Loading NF4 Transformer: {model_path}")
        # NO quantization_config for NF4 (quantization is baked in)
    elif is_fp8:
        print(f"Loading FP8 Transformer: {model_path}")
        # NO quantization_config for FP8
    else:
        print(f"Loading 4-bit BNB Transformer: {model_path}")
        transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_path,
        **transformer_load_kwargs
    ).to("cuda") # Move transformer manually for all cases for now

    print("✅ Transformer loaded!")
    allocated_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(VRAM allocated: {allocated_mb:.2f} MB)")

    # --- 3. Load Pipeline ---
    print(f"Loading Pipeline definition for: {model_path}")
    # Instantiate the correct scheduler
    scheduler = scheduler_class(
        num_train_timesteps=1000, # Assuming constant, adjust if needed
        shift=config["shift"],
        use_dynamic_shifting=False # Assuming constant
    )

    # Load pipeline components WITHOUT reloading models
    pipe = HiDreamImagePipeline.from_pretrained(
        model_path, # Load config/etc from the correct path
        scheduler=scheduler,
        tokenizer_4=tokenizer,
        text_encoder_4=text_encoder, # Pass the already loaded encoder
        transformer=None, # Pass None initially, will overwrite below
        torch_dtype=model_dtype,
        # If using device_map="auto" for text_encoder, pipeline might need manual placement?
        # Let's try moving the whole pipe at the end.
        low_cpu_mem_usage=True # Add for pipeline loading too
    )

    # --- 4. Final Setup ---
    print("Assigning explicitly loaded transformer to pipeline.")
    pipe.transformer = transformer # Crucial: ensure correct transformer is used

    print("Moving pipeline to CUDA...")
    pipe.to("cuda") # Move entire pipeline object

    if is_nf4:
        print("Enabling sequential CPU offload for NF4 model.")
        try:
            pipe.enable_sequential_cpu_offload()
            print("✅ Sequential CPU offload enabled.")
        except AttributeError:
            print("⚠️ Warning: enable_sequential_cpu_offload() not available on this pipeline version?")
        except Exception as e:
            print(f"⚠️ Warning: Failed to enable sequential CPU offload: {e}")


    print("✅ Pipeline ready!")
    allocated_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(VRAM allocated: {allocated_mb:.2f} MB)")

    return pipe, config


# Parse resolution string (keep simple version for now)
def parse_resolution(resolution_str):
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

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    _model_cache = {} # Keep original caching

    @classmethod
    def INPUT_TYPES(s):
        # Ensure options are ordered nicely if desired, or just use keys
        model_type_options = list(MODEL_CONFIGS.keys())
        # Set a reasonable default, maybe 'fast-nf4' if available
        default_model = "fast-nf4" if "fast-nf4" in model_type_options else \
                        "fast" if "fast" in model_type_options else model_type_options[0]

        return {
            "required": {
                "model_type": (model_type_options, {"default": default_model}), # Includes NF4
                "prompt": ("STRING", {"multiline": True, "default": "A photo of an astronaut riding a horse on the moon"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg):

        # Load or retrieve cached model
        # Use a lock maybe if concurrency becomes an issue, but unlikely in standard ComfyUI
        if model_type not in self._model_cache:
            # Clear potentially large unused models before loading a new one
            # Note: This simple cache doesn't track usage, just existence.
            # A more sophisticated cache (LRU) could be better but adds complexity.
            if self._model_cache:
                 print(f"Clearing cache before loading {model_type}...")
                 # Explicitly delete references and clear cache
                 keys_to_del = list(self._model_cache.keys())
                 for key in keys_to_del:
                     try:
                         pipe_to_del, _ = self._model_cache.pop(key)
                         del pipe_to_del # Hint GC
                     except Exception as del_e:
                         print(f"Error deleting cached model {key}: {del_e}")
                 gc.collect()
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()


            print(f"Loading model for {model_type}...")
            try:
                # Call the updated loading function
                pipe, config = load_models(model_type)
                self._model_cache[model_type] = (pipe, config)
                print(f"Model for {model_type} cached successfully!")
            except Exception as e:
                print(f"!!! ERROR loading model {model_type}: {e}")
                if model_type in self._model_cache: del self._model_cache[model_type]
                import traceback
                traceback.print_exc()
                raise e # Propagate error
        else:
            print(f"Using cached model for {model_type}")
            pipe, config = self._model_cache[model_type]

        # --- Get Config & Parse Inputs ---
        height, width = parse_resolution(resolution)
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed ---
        device = pipe.device
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"[HiDream Node] Starting generation: Model={model_type}, H={height}, W={width}, Steps={num_inference_steps}, CFG={guidance_scale}, Seed={seed}")
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        def progress_callback(step, timestep, latents): pbar.update(1)

        # --- Run Inference ---
        output_images = None
        try:
             # Use inference_mode for potential memory savings during the diffusion loop
             with torch.inference_mode():
                 output_images = pipe(
                     prompt=prompt,
                     height=height,
                     width=width,
                     guidance_scale=guidance_scale,
                     num_inference_steps=num_inference_steps,
                     num_images_per_prompt=1,
                     generator=generator,
                     callback_steps = 1,
                     callback = progress_callback,
                 ).images
        except Exception as e:
             print(f"!!! ERROR during pipeline execution: {e}")
             import traceback
             traceback.print_exc()
             # Return blank image on error
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)
        finally:
            pbar.update_absolute(num_inference_steps) # Ensure bar completes


        print("[HiDream Node] Generation Complete.")

        # --- Convert to ComfyUI Tensor ---
        if not output_images:
             print("[HiDream Node] ERROR: No images were generated.")
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)

        output_tensor = pil2tensor(output_images[0])

        # Model cleanup is handled by cache logic or ComfyUI's overall management

        return (output_tensor,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler (NF4/FP8/BNB)" # Updated name
}
