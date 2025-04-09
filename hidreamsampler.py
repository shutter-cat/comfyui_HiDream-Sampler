# -*- coding: utf-8 -*-
# HiDream Sampler Node for ComfyUI
# Version: 2024-07-29b (NF4/FP8/BNB Support, Indentation Fix)
#
# Required Dependencies:
# - transformers, diffusers, torch, numpy, Pillow
# - For NF4 models: optimum, accelerate, auto-gptq (`pip install optimum accelerate auto-gptq`)
# - For non-NF4/FP8 models (4-bit): bitsandbytes (`pip install bitsandbytes`)
# - Ensure hi_diffusers library is locally available or hdi1 package is installed.

import torch
import numpy as np
from PIL import Image
import comfy.utils
import gc
import os # For checking paths if needed

# --- Optional Dependency Handling ---
try:
    import accelerate
    accelerate_available = True
except ImportError:
    accelerate_available = False
    print("Warning: accelerate not installed. device_map='auto' for GPTQ models will not be available.")

try:
    import auto_gptq
    autogptq_available = True
except ImportError:
    autogptq_available = False
    # Note: Optimum might still load GPTQ without auto-gptq if using ExLlama kernels,
    # but it's often required. Add a warning if NF4 models are selected later.

try:
    import optimum
    optimum_available = True
except ImportError:
    optimum_available = False
    # Add a warning if NF4 models are selected later.

try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    # This case was handled before, just confirm variable state
    bnb_available = False
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")

# --- Core Imports ---
from transformers import LlamaForCausalLM, AutoTokenizer # Use AutoTokenizer

# --- HiDream Specific Imports ---
# Attempt local import first, then fallback (which might fail)
try:
    # Assuming hi_diffusers is cloned into this custom_node's directory
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    hidream_classes_loaded = True
except ImportError as e:
    print("--------------------------------------------------------------------")
    print(f"ComfyUI-HiDream-Sampler: Could not import local hi_diffusers ({e}).")
    print("Please ensure hi_diffusers library is inside ComfyUI-HiDream-Sampler,")
    print("or hdi1 package is installed in the ComfyUI environment.")
    print("Node may fail to load models.")
    print("--------------------------------------------------------------------")
    # Define placeholders so the script doesn't crash immediately
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None
    FlowUniPCMultistepScheduler = None
    FlashFlowMatchEulerDiscreteScheduler = None
    hidream_classes_loaded = False


# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
FP8_MODEL_PATH = "shuttleai/HiDream-I1-Full-FP8" # Specific path for FP8 version

ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4

# --- Model Configurations ---
# Added flags for dependency checks
MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False, "requires_gptq_deps": True
    },
     # --- FP8 Model ---
    "full-fp8": {
        "path": FP8_MODEL_PATH,
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": True, "requires_bnb": True, "requires_gptq_deps": False # LLM still uses BNB
    },
    # --- Original/BNB Models ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0, "num_inference_steps": 50, "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0, "num_inference_steps": 28, "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0, "num_inference_steps": 16, "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True, "requires_gptq_deps": False
    }
}

# --- Filter models based on available dependencies ---
original_model_count = len(MODEL_CONFIGS)
if not bnb_available:
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)}
if not optimum_available or not autogptq_available: # Require both for GPTQ
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_gptq_deps", False)}
if not hidream_classes_loaded: # If core classes failed, disable all
     MODEL_CONFIGS = {}

filtered_model_count = len(MODEL_CONFIGS)
if filtered_model_count == 0:
     print("*"*70)
     print("CRITICAL ERROR: No HiDream models available for HiDream Sampler node.")
     print("Check dependencies (bitsandbytes, optimum, accelerate, auto-gptq)")
     print("and ensure hi_diffusers library/hdi1 package is accessible.")
     print("*"*70)
elif filtered_model_count < original_model_count:
     print("*"*70)
     print("Warning: Some HiDream models disabled due to missing dependencies.")
     print(f"Available models: {list(MODEL_CONFIGS.keys())}")
     print("Check: bitsandbytes, optimum, accelerate, auto-gptq")
     print("*"*70)


# Define BitsAndBytes configs (if available)
bnb_llm_config = None
bnb_transformer_4bit_config = None
if bnb_available:
    bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

model_dtype = torch.bfloat16 # Keep bfloat16

# Get available scheduler classes from this module's scope
# Ensure these names match the actual imported class names
available_schedulers = {}
if hidream_classes_loaded:
    available_schedulers = {
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler
    }

# --- Helper: Get Scheduler Instance ---
def get_scheduler_instance(scheduler_name, shift_value):
    if not available_schedulers:
         raise RuntimeError("No schedulers available, HiDream classes failed to load.")
    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(f"Scheduler class '{scheduler_name}' not found or not loaded.")
    return scheduler_class(
        num_train_timesteps=1000,
        shift=shift_value,
        use_dynamic_shifting=False
    )

# --- Loading Function (Handles NF4, FP8, and default BNB) ---
def load_models(model_type):
    # Double-check core classes loaded
    if not hidream_classes_loaded:
        raise ImportError("Cannot load models: HiDream custom pipeline/transformer classes failed to import.")

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    is_fp8 = config.get("is_fp8", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)
    requires_gptq_deps = config.get("requires_gptq_deps", False)

    # Check dependencies again before attempting load
    if requires_bnb and not bnb_available:
         raise ImportError(f"Model type '{model_type}' requires BitsAndBytes but it's not installed.")
    if requires_gptq_deps and (not optimum_available or not autogptq_available):
         raise ImportError(f"Model type '{model_type}' requires Optimum & AutoGPTQ but they are not installed.")

    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, FP8: {is_fp8}, Requires BNB: {requires_bnb}, Requires GPTQ deps: {requires_gptq_deps}")
    start_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(Start VRAM: {start_mem:.2f} MB)")

    # --- 1. Load LLM (Conditional) ---
    text_encoder_load_kwargs = {
        "output_hidden_states": True, "low_cpu_mem_usage": True, "torch_dtype": model_dtype,
    }
    if is_nf4:
        llama_model_name = NF4_LLAMA_MODEL_NAME
        print(f"\n[1a] Preparing to load NF4-compatible LLM (GPTQ): {llama_model_name}")
        if accelerate_available: text_encoder_load_kwargs["device_map"] = "auto"; print("     Using device_map='auto' (requires accelerate).")
        else: print("     accelerate not found, will attempt manual CUDA placement.")
    else:
        llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
        print(f"\n[1a] Preparing to load Standard LLM (4-bit BNB): {llama_model_name}")
        if bnb_llm_config: text_encoder_load_kwargs["quantization_config"] = bnb_llm_config; print("     Using 4-bit BNB quantization.")
        else: raise ImportError("BNB config required for standard LLM but unavailable.")
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"

    print(f"[1b] Loading Tokenizer: {llama_model_name}...")
    try:
         tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    except Exception as e:
         print(f"Error loading tokenizer {llama_model_name}: {e}")
         raise
    print("     Tokenizer loaded.")

    print(f"[1c] Loading Text Encoder: {llama_model_name}...")
    print("     (This may take time and download files...)")
    try:
         text_encoder = LlamaForCausalLM.from_pretrained(llama_model_name, **text_encoder_load_kwargs)
    except Exception as e:
         print(f"Error loading text encoder {llama_model_name}: {e}")
         # (Add hints as before)
         if "gated repo" in str(e) or "401 Client Error" in str(e): print("Hint: Check HF login/license for gated model.")
         if "optimum" in str(e) and not optimum_available: print("Hint: Ensure `optimum` is installed.")
         if "auto-gptq" in str(e) and not autogptq_available: print("Hint: Ensure `auto-gptq` is installed.")
         raise

    if "device_map" not in text_encoder_load_kwargs:
        print("     Moving text encoder to CUDA...")
        try:
            text_encoder.to("cuda")
        except Exception as e:
            print(f"     Error moving text encoder to CUDA: {e}. Check model state.")
            raise

    step1_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Text encoder loaded! (VRAM: {step1_mem:.2f} MB)")

    # --- 2. Load Transformer (Conditional) ---
    print(f"\n[2] Preparing to load Diffusion Transformer from: {model_path}")
    transformer_load_kwargs = {
        "subfolder": "transformer", "torch_dtype": model_dtype, "low_cpu_mem_usage": True
    }
    if is_nf4:
        print("     Type: NF4 (Quantization included in model files)")
    elif is_fp8:
        print("     Type: FP8 (Quantization included in model files)")
        # transformer_load_kwargs["variant"] = "fp8" # Uncomment if needed
    else: # Default BNB case
        print("     Type: Standard (Applying 4-bit BNB quantization)")
        # *** CORRECTED INDENTATION HERE ***
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else:
            raise ImportError("BNB config required for transformer but unavailable.")

    print("     Loading Transformer model...")
    print("     (This may take time and download files...)")
    try:
        transformer = HiDreamImageTransformer2DModel.from_pretrained(model_path, **transformer_load_kwargs)
        print("     Moving Transformer to CUDA...")
        transformer.to("cuda") # Move transformer manually
    except Exception as e:
        print(f"Error loading/moving transformer {model_path}: {e}")
        raise

    step2_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Transformer loaded! (VRAM: {step2_mem:.2f} MB)")

    # --- 3. Load Scheduler ---
    print(f"\n[3] Preparing Scheduler: {scheduler_name}")
    scheduler = get_scheduler_instance(scheduler_name, shift)
    print(f"     Using Scheduler: {scheduler_name}")

    # --- 4. Load Pipeline (Passing Pre-loaded Components) ---
    print(f"\n[4] Loading Pipeline definition from: {model_path}")
    print("     Passing pre-loaded Scheduler, Tokenizer, Text Encoder...")
    try:
        # *** Pass loaded components directly ***
        pipe = HiDreamImagePipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            tokenizer_4=tokenizer,
            text_encoder_4=text_encoder,
            transformer=None, # Still pass None initially
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        print("     Pipeline structure loaded, using pre-loaded components.")

    except Exception as e:
        print(f"Error loading pipeline definition {model_path}: {e}")
        raise

    # --- 5. Final Pipeline Setup ---
    print("\n[5] Finalizing Pipeline Setup...")
    print("     Assigning loaded transformer to pipeline...")
    pipe.transformer = transformer

    print("     Moving pipeline object to CUDA (final check)...")
    try:
        pipe.to("cuda")
    except Exception as e:
        print(f"     Warning: Could not move pipeline object to CUDA: {e}.")

    if is_nf4: # (Same as before)
        print("     Attempting to enable sequential CPU offload for NF4 model...")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try: pipe.enable_sequential_cpu_offload(); print("     ✅ Sequential CPU offload enabled.")
            except Exception as e: print(f"     ⚠️ Warning: Failed to enable sequential CPU offload: {e}")
        else: print("     ⚠️ Warning: enable_sequential_cpu_offload() method not found on pipeline.")

    final_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"✅ Pipeline ready! (VRAM: {final_mem:.2f} MB)")

    return pipe, config


# --- Resolution Parsing & Tensor Conversion ---
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

def parse_resolution(resolution_str):
    """Parses resolution string like '1024 × 1024 (Square)' into (height, width)."""
    try:
        res_part = resolution_str.split(" (")[0]
        # Handle potential unicode multiplication sign vs 'x'
        parts = res_part.replace('x', '×').split("×")
        if len(parts) != 2:
             raise ValueError("Resolution string format incorrect.")
        w_str, h_str = [p.strip() for p in parts]
        width = int(w_str)
        height = int(h_str)
        #print(f"[HiDream Node] Parsed Resolution: Width={width}, Height={height}")
        # Pipeline takes height, width arguments
        return height, width # Return Height, Width
    except Exception as e:
        print(f"[HiDream Node] Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024 # Return default H, W

def pil2tensor(image: Image.Image):
    """Converts a PIL Image to a ComfyUI-compatible tensor."""
    if image is None: return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    _model_cache = {} # Simple cache

    @classmethod
    def INPUT_TYPES(s):
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
             # Provide a dummy input if no models are compatible
             return {"required": {"error": ("STRING", {"default": "No models available. Check dependencies (BNB/Optimum/AutoGPTQ/Accelerate) and logs.", "multiline": True})}}

        default_model = "fast-nf4" if "fast-nf4" in available_model_types else \
                        "fast" if "fast" in available_model_types else available_model_types[0]

        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
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

    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg, **kwargs):

        # Handle case where no models were available from INPUT_TYPES
        if not MODEL_CONFIGS or model_type == "error":
             print("HiDream Sampler Error: Node cannot operate, no compatible models found or loaded.")
             blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32) # Use default size
             return (blank_image,)

        pipe = None
        config = None
        # --- Model Loading / Caching ---
        # (Keep the caching logic the same as the previous version)
        if model_type in self._model_cache:
            print(f"Checking cached model for {model_type}...")
            pipe, config = self._model_cache[model_type]
            valid_cache = True
            if pipe is None or config is None or not hasattr(pipe, 'transformer') or pipe.transformer is None: # Added config check
                 valid_cache = False; print(f"Cached model for {model_type} seems invalid/unloaded. Reloading...")
                 if model_type in self._model_cache: del self._model_cache[model_type]
                 pipe, config = None, None # Force reload
            if valid_cache: print(f"Using valid cached model for {model_type}.")

        if pipe is None:
             if self._model_cache:
                  print(f"Clearing ALL cached models before loading {model_type}...")
                  # (Keep cache clearing logic the same)
                  keys_to_del = list(self._model_cache.keys())
                  for key in keys_to_del: # ... (same loop) ...
                      print(f"  Removing '{key}' from cache...")
                      try: # ... (same try/except) ...
                          pipe_to_del, _ = self._model_cache.pop(key)
                          if hasattr(pipe_to_del, 'transformer'): del pipe_to_del.transformer
                          if hasattr(pipe_to_del, 'text_encoder_4'): del pipe_to_del.text_encoder_4
                          del pipe_to_del
                      except Exception as del_e: print(f"  Error deleting cached model components for {key}: {del_e}")
                  gc.collect(); # Force GC
                  if torch.cuda.is_available(): torch.cuda.empty_cache()
                  print("Cache cleared.")

             print(f"Loading model for {model_type}...")
             try: # ... (same try/except) ...
                 pipe, config = load_models(model_type)
                 self._model_cache[model_type] = (pipe, config)
                 print(f"Model for {model_type} loaded and cached successfully!")
             except Exception as e: # ... (same error handling) ...
                 print(f"!!! ERROR loading model {model_type}: {e}")
                 if model_type in self._model_cache: del self._model_cache[model_type]
                 import traceback; traceback.print_exc()
                 blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                 return (blank_image,)

        # --- Generation Setup ---
        if pipe is None or config is None: # Ensure config was loaded
            print("CRITICAL ERROR: Pipeline or config is None after loading attempt.")
            blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_image,)

        # *** DEFINE variables needed BEFORE the try block ***
        is_nf4_current = config.get("is_nf4", False)
        height, width = parse_resolution(resolution) # Define height/width early
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]
        pbar = comfy.utils.ProgressBar(num_inference_steps) # Define pbar early
        def progress_callback(step, timestep, latents): pbar.update(1) # Define callback here too

        try: # Use comfy's helper to get the primary compute device
            inference_device = comfy.model_management.get_torch_device()
        except Exception as e:
             print(f"Warning: Could not get device via comfy.model_management ({e}). Falling back.")
             inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Creating Generator on device: {inference_device}")
        generator = torch.Generator(device=inference_device).manual_seed(seed)

        print(f"\n--- Starting Generation ---")
        print(f"Model: {model_type}, Res: {height}x{width}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")

        # --- Run Inference ---
        output_images = None
        try:
             if not is_nf4_current: # Use the correctly defined variable
                 print(f"Ensuring pipeline is on device: {inference_device} (Offload NOT enabled)")
                 pipe.to(inference_device)
             else:
                 print(f"Skipping pipe.to({inference_device}) because sequential CPU offload is enabled.")

             with torch.inference_mode():
                 # Removed callback_steps=1
                 output_images = pipe(
                     prompt=prompt,
                     height=height,
                     width=width,
                     guidance_scale=guidance_scale,
                     num_inference_steps=num_inference_steps,
                     num_images_per_prompt=1,
                     generator=generator,
                     callback=progress_callback, # Keep callback
                 ).images
        except Exception as e: # Handle errors during inference
             print(f"!!! ERROR during pipeline execution: {e}")
             if isinstance(e, TypeError) and 'unexpected keyword argument' in str(e):
                  print("Hint: Check if the arguments passed to the pipe() call match the pipeline's definition.")
             import traceback; traceback.print_exc()
             # height/width are defined now, safe to use here
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)
        finally:
            # pbar is defined now, safe to use here
            pbar.update_absolute(num_inference_steps)

        print("--- Generation Complete ---")

        # --- Convert to ComfyUI Tensor ---
        if not output_images:
             print("[HiDream Node] ERROR: No images returned from pipeline.")
             # height/width are defined now
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)

        output_tensor = pil2tensor(output_images[0])

        return (output_tensor,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler (NF4/FP8/BNB)"
}

# --- Startup Print ---
print("-" * 50)
print("HiDream Sampler Node Initialized")
print(f"Available Models (after dependency checks): {list(MODEL_CONFIGS.keys())}")
print("-" * 50)
