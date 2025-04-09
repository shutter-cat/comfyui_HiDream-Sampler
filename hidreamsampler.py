import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm
import comfy.utils
import gc

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
        "is_fp8": False # Flag to indicate if it's FP8
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
        "scheduler_class": FlowUniPCMultistepScheduler,
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

# Define BitsAndBytes configs *outside* loading functions to avoid recreation
# We only need the LLM one now if we don't quantize the FP8 transformer
# bnb_transformer_config = DiffusersBitsAndBytesConfig(load_in_4bit=True) # Only for non-FP8
bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
model_dtype = torch.bfloat16 # bfloat16 is often used alongside FP8

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    parts = resolution_str.split(" (")[0].split(" × ")
    try:
        p_width = int(parts[0].strip())
        p_height = int(parts[1].strip())
        width = p_width
        height = p_height
        print(f"[HiDream Node] Parsed Resolution: Width={width}, Height={height}")
        return height, width # Return Height, Width
    except Exception as e:
        print(f"[HiDream Node] Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024

def pil2tensor(image: Image.Image):
    if image is None: return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    @classmethod
    def INPUT_TYPES(s):
        # Update the model_type list to include 'full-fp8'
        model_type_options = list(MODEL_CONFIGS.keys())
        default_model = "fast" if "fast" in model_type_options else model_type_options[0]

        return {
            "required": {
                # Use the updated list including 'full-fp8'
                "model_type": (model_type_options, {"default": default_model}),
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

        # --- Get Config ---
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model_type: {model_type}")
        config = MODEL_CONFIGS[model_type]
        hidream_model_path = config["path"]
        is_fp8_transformer = config.get("is_fp8", False) # Check if loading FP8 version

        # --- Parse Inputs ---
        height, width = parse_resolution(resolution)

        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed & Device ---
        device = mm.get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        # --- Initialize variables ---
        text_encoder, tokenizer, transformer, scheduler, pipe, prompt_embeds = None, None, None, None, None, None

        try:
            # --- Step 1: Load Text Encoder and Tokenizer ---
            print(f"[HiDream Node] Stage 1: Loading LLM Text Encoder ({LLAMA_MODEL_NAME})")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
            text_encoder = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                output_hidden_states=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_llm_config, # Quantize the LLM
                torch_dtype=model_dtype,
                attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
            )
            try:
                 text_encoder.to(device)
            except Exception as move_err:
                 print(f"[HiDream Node] Warning: Could not explicitly move text_encoder to {device}. Error: {move_err}")
                 if next(text_encoder.parameters()).device != device:
                      print(f"[HiDream Node] Error: Text encoder is on {next(text_encoder.parameters()).device} NOT target {device}!")

            print(f"[HiDream Node] Text Encoder Loaded. Memory: {mm.get_model_size(text_encoder)/1e9:.2f} GB")

            # --- Step 2: Encode Prompt ---
            print("[HiDream Node] Stage 2: Encoding Prompt")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.inference_mode():
                outputs = text_encoder(**inputs, output_hidden_states=True)
                prompt_embeds = outputs.hidden_states[-1].to(dtype=model_dtype, device=device).detach()
            print(f"[HiDream Node] Prompt Encoded. Shape: {prompt_embeds.shape}")

            # --- Step 3: Unload Text Encoder ---
            print("[HiDream Node] Stage 3: Unloading Text Encoder")
            del text_encoder; text_encoder = None
            del tokenizer; tokenizer = None
            gc.collect(); mm.soft_empty_cache()
            # torch.cuda.empty_cache() # Optional: More forceful clear
            print("[HiDream Node] Text Encoder Unloaded.")
            mm.log_current_system_memory_usage()

            # --- Step 4: Load Diffusion Model (Transformer) ---
            transformer_load_kwargs = {
                "subfolder": "transformer",
                "low_cpu_mem_usage": True,
                "torch_dtype": model_dtype,
                # FP8 Specific flags MIGHT be needed depending on transformers/diffusers version and the model structure
                # e.g., variant="fp8", device_map="auto"
                # We'll try without them first.
            }

            if is_fp8_transformer:
                print(f"[HiDream Node] Stage 4: Loading FP8 Diffusion Transformer from {hidream_model_path}")
                # DO NOT apply bnb_config for FP8 models
                # Potentially add specific FP8 args here if needed:
                # transformer_load_kwargs["variant"] = "fp8"
            else:
                print(f"[HiDream Node] Stage 4: Loading 4-bit Diffusion Transformer from {hidream_model_path}")
                # Apply 4-bit quantization for non-FP8 models
                transformer_load_kwargs["quantization_config"] = DiffusersBitsAndBytesConfig(load_in_4bit=True)

            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                hidream_model_path,
                **transformer_load_kwargs
            )

            # Manually move to device (important for FP8 if not using device_map)
            try:
                transformer.to(device)
            except Exception as move_err:
                 print(f"[HiDream Node] Warning: Could not explicitly move transformer to {device}. Error: {move_err}")
                 if next(transformer.parameters()).device != device:
                      print(f"[HiDream Node] Error: Transformer is on {next(transformer.parameters()).device} NOT target {device}!")

            print(f"[HiDream Node] Transformer Loaded. Memory: {mm.get_model_size(transformer)/1e9:.2f} GB")
            mm.log_current_system_memory_usage()


            # --- Step 5: Prepare Pipeline ---
            print("[HiDream Node] Stage 5: Preparing Pipeline")
            scheduler_class = config["scheduler_class"]
            scheduler = scheduler_class(num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
            pipe = HiDreamImagePipeline(transformer=transformer, scheduler=scheduler, tokenizer_4=None, text_encoder_4=None)
            print("[HiDream Node] Pipeline Ready.")

            # --- Step 6: Run Inference ---
            print("[HiDream Node] Stage 6: Running Diffusion Inference")
            output_images = None
            pbar = comfy.utils.ProgressBar(num_inference_steps)
            def progress_callback(step, timestep, latents): pbar.update(1)

            with torch.inference_mode():
                output_images = pipe(
                    prompt_embeds=prompt_embeds, prompt=None, height=height, width=width,
                    guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1, generator=generator, callback_steps=1, callback=progress_callback,
                ).images
            pbar.update_absolute(num_inference_steps)
            print("[HiDream Node] Generation Complete.")

        except Exception as e:
            print(f"!!! Exception during processing !!! {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("[HiDream Node] ERROR: Exception occurred. Returning blank image.")
            blank_image = torch.zeros((1, height if 'height' in locals() else 512, width if 'width' in locals() else 512, 3), dtype=torch.float32)
            return (blank_image,)

        finally:
            # --- Step 7: Clean up Diffusion Model ---
            print("[HiDream Node] Stage 7: Cleaning up Diffusion Model")
            del pipe; del transformer; del scheduler # Delete in reverse order of creation/dependency
            pipe, transformer, scheduler = None, None, None # Clear refs
            gc.collect(); mm.soft_empty_cache()
            # torch.cuda.empty_cache() # Optional
            print("[HiDream Node] Diffusion Model Unloaded.")
            mm.log_current_system_memory_usage()

            # --- Step 8: Convert to ComfyUI Tensor ---
            if output_images is None or len(output_images) == 0:
                 print("[HiDream Node] ERROR: No images generated. Returning blank image.")
                 blank_image = torch.zeros((1, height if 'height' in locals() else 512, width if 'width' in locals() else 512, 3), dtype=torch.float32)
                 return (blank_image,)

            output_tensor = pil2tensor(output_images[0])
            return (output_tensor,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler (Mem Optimized v3 FP8)" # Updated name
}
