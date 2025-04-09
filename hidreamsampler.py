# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
import comfy.utils
import gc

# Diffusers/Transformers imports
try:
    from .hi_diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
    from .hi_diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
    from .hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from .hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
except ImportError:
    print("--------------------------------------------------------------------")
    print("ComfyUI-HiDream-Sampler: Could not import local hi_diffusers.")
    print("Please ensure you have cloned the hi_diffusers library into")
    print("the ComfyUI-HiDream-Sampler custom node directory,")
    print("or installed the hdi1 package as per instructions.")
    print("Falling back to attempting global diffusers/transformers imports.")
    print("--------------------------------------------------------------------")
    # Fallback attempt - This might not work if the HiDream specific classes aren't discoverable
    from diffusers import DiffusionPipeline # Placeholder, might need specific pipeline import
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler as FlashFlowMatchEulerDiscreteScheduler # Guessing name
    from diffusers.schedulers import UniPCMultistepScheduler as FlowUniPCMultistepScheduler # Guessing name
    # The custom transformer and pipeline might not be available globally
    HiDreamImageTransformer2DModel = None
    HiDreamImagePipeline = None

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer

# Quantization configs (only needed for non-NF4/non-FP8)
try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    bnb_available = True
except ImportError:
    print("Warning: bitsandbytes not installed. 4-bit BNB quantization will not be available.")
    TransformersBitsAndBytesConfig = None
    DiffusersBitsAndBytesConfig = None
    bnb_available = False

# --- Model Paths ---
ORIGINAL_MODEL_PREFIX = "HiDream-ai"
NF4_MODEL_PREFIX = "azaneko"
FP8_MODEL_PATH = "shuttleai/HiDream-I1-Full-FP8" # Specific path for FP8 version

ORIGINAL_LLAMA_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" # For original/FP8
NF4_LLAMA_MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" # For NF4

# --- Model Configurations ---
MODEL_CONFIGS = {
    # --- NF4 Models ---
    "full-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Full-nf4",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler", # Use string names for dynamic import
        "is_nf4": True, "is_fp8": False, "requires_bnb": False
    },
    "dev-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Dev-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False
    },
    "fast-nf4": {
        "path": f"{NF4_MODEL_PREFIX}/HiDream-I1-Fast-nf4",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": True, "is_fp8": False, "requires_bnb": False
    },
     # --- FP8 Model ---
    "full-fp8": {
        "path": FP8_MODEL_PATH,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": True, "requires_bnb": True # LLM still uses BNB
    },
    # --- Original/BNB Models ---
     "full": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": "FlowUniPCMultistepScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True
    },
    "dev": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True
    },
    "fast": {
        "path": f"{ORIGINAL_MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": "FlashFlowMatchEulerDiscreteScheduler",
        "is_nf4": False, "is_fp8": False, "requires_bnb": True
    }
}

# Filter models if BNB is not available
if not bnb_available:
    MODEL_CONFIGS = {k: v for k, v in MODEL_CONFIGS.items() if not v.get("requires_bnb", False)}
    if not MODEL_CONFIGS:
        print("ERROR: No compatible HiDream models found. BNB is required for most models but is not installed.")
        # Optionally raise an error or just leave the dictionary empty
        # raise ImportError("BitsAndBytes is required for available HiDream models but not installed.")


# Define BitsAndBytes configs (if available)
bnb_llm_config = None
bnb_transformer_4bit_config = None
if bnb_available:
    bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    bnb_transformer_4bit_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)

model_dtype = torch.bfloat16 # Keep bfloat16

# Get available scheduler classes from this module's scope
available_schedulers = {
    "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
    "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler
}

# --- Helper: Get Scheduler Instance ---
def get_scheduler_instance(scheduler_name, shift_value):
    scheduler_class = available_schedulers.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(f"Scheduler class '{scheduler_name}' not found.")
    # Check if the class actually exists (was imported)
    if not scheduler_class:
         raise ImportError(f"Scheduler class '{scheduler_name}' definition is missing. Check imports.")
    return scheduler_class(
        num_train_timesteps=1000,
        shift=shift_value,
        use_dynamic_shifting=False
    )

# --- Loading Function (Handles NF4, FP8, and default BNB) ---
def load_models(model_type):
    if not HiDreamImagePipeline or not HiDreamImageTransformer2DModel:
        raise ImportError("HiDream custom pipeline/transformer classes not loaded. Cannot proceed.")

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown or incompatible model_type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model_path = config["path"]
    is_nf4 = config.get("is_nf4", False)
    is_fp8 = config.get("is_fp8", False)
    scheduler_name = config["scheduler_class"]
    shift = config["shift"]
    requires_bnb = config.get("requires_bnb", False)

    if requires_bnb and not bnb_available:
         raise ImportError(f"Model type '{model_type}' requires BitsAndBytes (BNB) for quantization, but it's not installed.")


    print(f"--- Loading Model Type: {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"NF4: {is_nf4}, FP8: {is_fp8}, Requires BNB: {requires_bnb}")

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
        # Check if accelerate is available
        try:
            import accelerate
            text_encoder_load_kwargs["device_map"] = "auto"
            print("Using device_map='auto' for GPTQ LLM.")
        except ImportError:
            print("Warning: accelerate not installed. Cannot use device_map='auto'. Attempting manual placement.")
            # Fallback or raise error? For now, we'll let manual placement handle it later.
    else:
        llama_model_name = ORIGINAL_LLAMA_MODEL_NAME
        print(f"Loading Standard LLM (4-bit BNB): {llama_model_name}")
        if bnb_llm_config:
             text_encoder_load_kwargs["quantization_config"] = bnb_llm_config
        text_encoder_load_kwargs["attn_implementation"] = "flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"

    print(f"Loading Tokenizer: {llama_model_name}")
    try:
        # Use AutoTokenizer for better flexibility with different model types
        tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_fast=False)
    except Exception as e:
        print(f"Error loading tokenizer {llama_model_name}: {e}")
        raise

    print(f"Loading Text Encoder: {llama_model_name}")
    try:
        text_encoder = LlamaForCausalLM.from_pretrained(
            llama_model_name,
            **text_encoder_load_kwargs
        )
    except Exception as e:
        print(f"Error loading text encoder {llama_model_name}: {e}")
        # Add hint about Hugging Face login for gated models like Llama
        if "gated repo" in str(e) or "401 Client Error" in str(e):
             print("--------------------------------------------------------------------")
             print("Hint: This may be a gated model. Ensure you have accepted the")
             print("license on Hugging Face Hub and are logged in via")
             print("`huggingface-cli login` in your ComfyUI environment.")
             print("--------------------------------------------------------------------")
        raise

    # Manually move if device_map wasn't used
    if "device_map" not in text_encoder_load_kwargs:
        print("Moving text encoder to CUDA...")
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
    elif is_fp8:
        print(f"Loading FP8 Transformer: {model_path}")
        # May need variant="fp8" depending on library versions
        # transformer_load_kwargs["variant"] = "fp8"
    else: # Default BNB case
        print(f"Loading 4-bit BNB Transformer: {model_path}")
        if bnb_transformer_4bit_config:
            transformer_load_kwargs["quantization_config"] = bnb_transformer_4bit_config
        else:
             # Should have been caught earlier, but double-check
             raise ImportError("BNB config required for transformer but unavailable.")

    try:
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            model_path,
            **transformer_load_kwargs
        ).to("cuda") # Move transformer manually
    except Exception as e:
        print(f"Error loading transformer {model_path}: {e}")
        raise

    print("✅ Transformer loaded!")
    allocated_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(VRAM allocated: {allocated_mb:.2f} MB)")

    # --- 3. Load Pipeline ---
    print(f"Loading Pipeline definition for: {model_path}")
    scheduler = get_scheduler_instance(scheduler_name, shift)

    try:
        # Load pipeline structure, then assign components
        pipe = HiDreamImagePipeline.from_pretrained(
            model_path, # Load non-model files (configs etc) from here
            # Pass None for components we load manually to avoid reloading
            scheduler=None,
            tokenizer_4=None,
            text_encoder_4=None,
            transformer=None,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        # Assign pre-loaded components
        pipe.scheduler = scheduler
        pipe.tokenizer_4 = tokenizer
        pipe.text_encoder_4 = text_encoder
        # Leave transformer assignment until later

    except Exception as e:
        print(f"Error loading pipeline definition {model_path}: {e}")
        raise

    # --- 4. Final Setup ---
    print("Assigning explicitly loaded transformer to pipeline.")
    pipe.transformer = transformer # Crucial: ensure correct transformer is used

    # Move the whole pipeline object to CUDA if its components aren't already there
    # This helps ensure internal tensors/buffers are placed correctly
    # Note: device_map='auto' on text_encoder might make this tricky. Test carefully.
    print("Moving pipeline object to CUDA (final check)...")
    try:
        pipe.to("cuda")
    except Exception as e:
        print(f"Warning: Could not move entire pipeline object to CUDA. Components might be mixed device? Error: {e}")


    if is_nf4:
        print("Attempting to enable sequential CPU offload for NF4 model.")
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                print("✅ Sequential CPU offload enabled.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to enable sequential CPU offload: {e}")
        else:
             print("⚠️ Warning: enable_sequential_cpu_offload() method not found on pipeline.")

    print("✅ Pipeline ready!")
    allocated_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"(VRAM allocated: {allocated_mb:.2f} MB)")

    return pipe, config

# Resolution options - DEFINITION MOVED HERE, BEFORE THE CLASS
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Parse resolution string (keep simple version for now)
def parse_resolution(resolution_str):
    try:
        res_part = resolution_str.split(" (")[0]
        w_str, h_str = [p.strip() for p in res_part.split("×")]
        width = int(w_str)
        height = int(h_str)
        # The pipeline takes height, width arguments
        print(f"[HiDream Node] Parsed Resolution: Width={width}, Height={height}")
        return height, width # Return Height, Width
    except Exception as e:
        print(f"[HiDream Node] Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024

def pil2tensor(image: Image.Image):
    """Converts a PIL Image to a ComfyUI-compatible tensor."""
    if image is None: return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    _model_cache = {} # Keep original caching

    @classmethod
    def INPUT_TYPES(s):
        # Get available model types after filtering based on dependencies
        available_model_types = list(MODEL_CONFIGS.keys())
        if not available_model_types:
             print("CRITICAL ERROR: No HiDream models available based on installed dependencies.")
             # Provide a dummy input to prevent ComfyUI from completely failing
             return {"required": {"error": ("STRING", {"default": "No models available. Check dependencies (BNB/Accelerate) and logs.", "multiline": True})}}


        # Set a reasonable default
        default_model = "fast-nf4" if "fast-nf4" in available_model_types else \
                        "fast" if "fast" in available_model_types else available_model_types[0]

        return {
            "required": {
                "model_type": (available_model_types, {"default": default_model}),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of an astronaut riding a horse on the moon"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}), # Now RESOLUTION_OPTIONS is defined
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}),
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def generate(self, model_type, prompt, resolution, seed, override_steps, override_cfg, **kwargs): # Added **kwargs to catch potential 'error' input

        # Handle case where no models were available
        if model_type == "error" or not MODEL_CONFIGS:
             print("HiDream Sampler Error: No compatible models loaded. Check logs and dependencies.")
             # Return a blank image to avoid crashing the workflow
             blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
             return (blank_image,)


        # Load or retrieve cached model
        pipe = None
        config = None
        if model_type in self._model_cache:
            print(f"Using cached model for {model_type}")
            pipe, config = self._model_cache[model_type]
            # Quick check if pipe seems valid (e.g., still has device attribute)
            if pipe is None or not hasattr(pipe, 'device'):
                 print(f"Cached model for {model_type} seems invalid. Reloading...")
                 if model_type in self._model_cache: del self._model_cache[model_type] # Remove bad entry
                 pipe, config = None, None # Force reload

        if pipe is None:
             # Clear potentially large unused models before loading a new one
             if self._model_cache:
                  print(f"Clearing cache before loading {model_type}...")
                  keys_to_del = list(self._model_cache.keys())
                  for key in keys_to_del:
                      try:
                          pipe_to_del, _ = self._model_cache.pop(key)
                          del pipe_to_del
                      except Exception as del_e:
                          print(f"Error deleting cached model {key}: {del_e}")
                  gc.collect()
                  if torch.cuda.is_available(): torch.cuda.empty_cache()

             print(f"Loading model for {model_type}...")
             try:
                 pipe, config = load_models(model_type)
                 self._model_cache[model_type] = (pipe, config)
                 print(f"Model for {model_type} cached successfully!")
             except Exception as e:
                 print(f"!!! ERROR loading model {model_type}: {e}")
                 if model_type in self._model_cache: del self._model_cache[model_type]
                 import traceback
                 traceback.print_exc()
                 # Return blank image on loading error
                 blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32) # Use default size
                 return (blank_image,)


        # --- Get Config & Parse Inputs ---
        # Use config retrieved from cache/load
        if config is None: # Should not happen if loading succeeded, but as a safeguard
            config = MODEL_CONFIGS[model_type]

        height, width = parse_resolution(resolution)
        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed ---
        try:
             device = pipe.device
        except AttributeError:
             print("Error: Pipeline object has no device attribute. Assuming CUDA.")
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)

        print(f"[HiDream Node] Starting generation: Model={model_type}, H={height}, W={width}, Steps={num_inference_steps}, CFG={guidance_scale}, Seed={seed}")
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        def progress_callback(step, timestep, latents): pbar.update(1)

        # --- Run Inference ---
        output_images = None
        try:
             # Ensure pipe is on the correct device before inference
             pipe.to(device)
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
             # Return blank image on runtime error
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,)
        finally:
            pbar.update_absolute(num_inference_steps)


        print("[HiDream Node] Generation Complete.")

        # --- Convert to ComfyUI Tensor ---
        if not output_images:
             print("[HiDream Node] ERROR: No images were generated.")
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

print("-----------------------------------")
print("HiDream Sampler Node Loaded.")
print(f"Available Models: {list(MODEL_CONFIGS.keys())}")
print("-----------------------------------")
