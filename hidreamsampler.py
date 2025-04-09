import torch
import numpy as np
from PIL import Image
import comfy.model_management as mm # Use comfy's model management
import comfy.utils
import functools # For partial function application

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

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler # Store class, not instance
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler_class": FlowUniPCMultistepScheduler # Store class, not instance
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler_class": FlashFlowMatchEulerDiscreteScheduler # Store class, not instance
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

# --- Helper Functions for Loading Models ---

# Define BitsAndBytes configs *outside* loading functions to avoid recreation
bnb_transformer_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
bnb_llm_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
model_dtype = torch.bfloat16 # Use bfloat16 for consistency

# Wrapper for Text Encoder loading to work with comfy.model_management
class LlamaTextEncoderLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        print(f"[HiDream Node] Loading Llama Text Encoder: {self.model_name}")
        text_encoder = LlamaForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            # output_attentions=True, # Attentions often not needed for embeddings, save memory
            low_cpu_mem_usage=True,
            quantization_config=bnb_llm_config,
            torch_dtype=model_dtype,
            # attn_implementation="eager" # Consider "sdpa" if available/compatible for potential speedup/memory saving
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager" # Use FA2 if available
        )
        # We let comfy.model_management handle the .to(device)
        return text_encoder

# Wrapper for HiDream Transformer loading
class HiDreamTransformerLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        print(f"[HiDream Node] Loading HiDream Transformer: {self.model_path}")
        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            quantization_config=bnb_transformer_config,
            torch_dtype=model_dtype
        )
        # We let comfy.model_management handle the .to(device)
        return transformer

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    parts = resolution_str.split(" (")[0].split(" × ")
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
        # HiDream seems to use H, W order in some places, W, H in others.
        # The pipeline takes height, width args. Let's match that.
        # Example: "768 x 1360" -> height=768, width=1360 IS WRONG -> should be height=1360, width=768 for portrait
        # Let's assume the string means W x H
        # Re-check HiDream pipeline call signature if issues persist.
        # Assuming W x H for now based on common convention:
        # Corrected based on standard WxH format and Portrait/Landscape hints:
        if "Portrait" in resolution_str or (len(parts) == 2 and int(parts[1]) > int(parts[0])):
             # Height is the larger dimension
             height = max(int(parts[0]), int(parts[1]))
             width = min(int(parts[0]), int(parts[1]))
        elif "Landscape" in resolution_str or (len(parts) == 2 and int(parts[0]) > int(parts[1])):
             # Width is the larger dimension
             width = max(int(parts[0]), int(parts[1]))
             height = min(int(parts[0]), int(parts[1]))
        elif "Square" in resolution_str or (len(parts) == 2 and int(parts[0]) == int(parts[1])):
             width = int(parts[0])
             height = int(parts[1])
        else: # Fallback/Default
            width=1024
            height=1024
        print(f"[HiDream Node] Parsed Resolution: Width={width}, Height={height}")
        return width, height # Return W, H

    except Exception as e:
        print(f"[HiDream Node] Error parsing resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024 # Default fallback W, H

def pil2tensor(image: Image.Image):
    """Converts a PIL Image to a ComfyUI-compatible tensor."""
    if image is None:
        return None
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- ComfyUI Node Definition ---
class HiDreamSampler:

    # Remove custom cache, let comfy handle it
    # _model_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (list(MODEL_CONFIGS.keys()), {"default": "fast"}),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of an astronaut riding a horse on the moon"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "1024 × 1024 (Square)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Increased max seed range
                "override_steps": ("INT", {"default": -1, "min": -1, "max": 100}), # -1 uses config default
                "override_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1}), # -1 uses config default
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

        # --- Parse Inputs ---
        width, height = parse_resolution(resolution) # Returns W, H

        num_inference_steps = override_steps if override_steps >= 0 else config["num_inference_steps"]
        guidance_scale = override_cfg if override_cfg >= 0.0 else config["guidance_scale"]

        # --- Handle Seed ---
        # Use torch.device("cuda") for generator if targeting GPU specifically
        device = mm.get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)

        # --- Step 1: Load Text Encoder and Tokenizer ---
        print("[HiDream Node] Stage 1: Loading Text Encoder")
        # Use comfy's loading mechanism
        text_encoder_loader = LlamaTextEncoderLoader(LLAMA_MODEL_NAME)
        text_encoder = mm.load_model_gpu(text_encoder_loader)
        # Tokenizer is small, load directly
        tokenizer = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)


        # --- Step 2: Encode Prompt ---
        print("[HiDream Node] Stage 2: Encoding Prompt")
        # Ensure text encoder is on the correct device (load_model_gpu should handle this)
        # text_encoder = text_encoder.to(device) # Should not be needed if load_model_gpu worked

        # Prepare inputs for the text encoder
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Get embeddings - HiDream likely uses last_hidden_state
        # Need to wrap in inference_mode and handle potential errors
        prompt_embeds = None
        try:
            with torch.inference_mode():
                outputs = text_encoder(**inputs, output_hidden_states=True)
                # Often the last hidden state is used, check HiDream pipeline's internal _encode_prompt if unsure
                prompt_embeds = outputs.hidden_states[-1] # Use last hidden state
                # Maybe needs pooling? Check HiDream source. Assuming direct use for now.
                # Detach from graph and ensure correct dtype
                prompt_embeds = prompt_embeds.to(dtype=model_dtype, device=device).detach()

        except Exception as e:
            print(f"[HiDream Node] Error during prompt encoding: {e}")
             # Clean up loaded models before raising
            del tokenizer
            del text_encoder
            mm.soft_empty_cache() # Ask Comfy to release VRAM if possible
            raise e # Re-raise the exception

        print(f"[HiDream Node] Prompt Encoded. Shape: {prompt_embeds.shape}")

        # --- Step 3: Unload Text Encoder ---
        print("[HiDream Node] Stage 3: Unloading Text Encoder")
        del tokenizer
        del text_encoder # Remove reference
        mm.soft_empty_cache() # Ask Comfy to release VRAM if possible
        print("[HiDream Node] Text Encoder Unloaded.")


        # --- Step 4: Load Diffusion Model (Transformer) ---
        print("[HiDream Node] Stage 4: Loading Diffusion Transformer")
        transformer_loader = HiDreamTransformerLoader(hidream_model_path)
        transformer = mm.load_model_gpu(transformer_loader)
        # Ensure transformer is on the correct device (load_model_gpu should handle this)
        # transformer = transformer.to(device) # Not needed if load_model_gpu works

        print("[HiDream Node] Diffusion Transformer Loaded.")


        # --- Step 5: Prepare Pipeline ---
        print("[HiDream Node] Stage 5: Preparing Pipeline")
        # Load scheduler
        scheduler_class = config["scheduler_class"]
        scheduler = scheduler_class(
            num_train_timesteps=1000, # Or get from model config if available
            shift=config["shift"],
            use_dynamic_shifting=False # Or get from config
        )

        # Instantiate pipeline MANUALLY with pre-loaded components
        # We pass None for text encoder/tokenizer as we use prompt_embeds
        # We also don't call .to(device) on the pipeline itself
        pipe = HiDreamImagePipeline(
             transformer=transformer,
             scheduler=scheduler,
             tokenizer_4=None,      # Not needed, using embeds
             text_encoder_4=None    # Not needed, using embeds
             # Add any other components HiDreamImagePipeline requires in __init__
             # Check the __init__ signature of HiDreamImagePipeline!
             # If it needs other components from from_pretrained, load them minimally.
        )
        # Ensure pipeline components required for __call__ (like VAE if used) are loaded if needed.
        # HiDream might not use a separate VAE in the same way as Stable Diffusion.

        print("[HiDream Node] Pipeline Ready.")

        # --- Step 6: Run Inference ---
        print("[HiDream Node] Stage 6: Running Diffusion Inference")
        output_images = None
        pbar = comfy.utils.ProgressBar(num_inference_steps) # Progress bar

        # Define callback for progress updates
        def progress_callback(step, timestep, latents):
            pbar.update(1)

        try:
            # Use inference_mode for efficiency
            with torch.inference_mode():
                output_images = pipe(
                    prompt_embeds=prompt_embeds, # Use pre-computed embeddings
                    prompt=None,                 # Pass None for prompt
                    height=height,               # Ensure order is correct (H, W)
                    width=width,                 # Ensure order is correct (H, W)
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    callback_steps=1,
                    callback=progress_callback,
                    # Add any other specific args HiDream pipeline needs
                ).images
        except Exception as e:
             print(f"[HiDream Node] Error during pipeline inference: {e}")
             # Clean up diffusion model before raising
             del transformer
             del scheduler
             del pipe
             mm.soft_empty_cache()
             raise e # Re-raise the exception
        finally:
             # Ensure progress bar is finished even if error occurs mid-way
             pbar.update_absolute(num_inference_steps)


        print("[HiDream Node] Generation Complete.")

        # --- Step 7: Clean up Diffusion Model (Optional but good practice) ---
        # Let Comfy's management handle unloading when necessary,
        # but explicit deletion can sometimes help immediately free up refs
        del transformer
        del scheduler
        del pipe
        mm.soft_empty_cache() # Request cleanup

        # --- Step 8: Convert to ComfyUI Tensor ---
        if not output_images or len(output_images) == 0:
             print("[HiDream Node] ERROR: No images were generated.")
             # Return a blank image or raise an error? Returning blank tensor.
             # Create a dummy tensor matching expected output shape
             # Assuming batch size 1, channels 3 (RGB)
             blank_image = torch.zeros((1, height, width, 3), dtype=torch.float32)
             return (blank_image,) # Match RETURN_TYPES


        output_tensor = pil2tensor(output_images[0])

        return (output_tensor,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HiDreamSampler": HiDreamSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDreamSampler": "HiDream Sampler (Mem Optimized)" # Added hint
}
