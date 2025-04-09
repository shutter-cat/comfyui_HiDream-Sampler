Forked from original https://github.com/lum3on/comfyui_HiDream-Sampler

## Added fp8(full only) and NF4 (Full/Dev/Fast) download and load support
## Added better memory handling
## Added more informative CLI output for TQDM

Many thanks to the folks who created this and set it up for Comfy, I just spent a few hours adding better support for consumer GPUs.

Full/Dev/Fast requires roughly 27GB VRAM
NF4 requires roughly 15GB VRAM

###### NOTE - fp8 support is still a WIP.



![image](https://github.com/user-attachments/assets/3d4e9bee-772b-4c57-84cb-b5a6da30efd5)

# HiDreamSampler for ComfyUI

A custom ComfyUI node for generating images using the HiDream AI model.

## Features
- Supports `full`, `dev`, and `fast` model types.
- Configurable resolution and inference steps.
- Uses 4-bit quantization for lower memory usage.

## Installation
Please make sure you have installed Flash Attention. We recommend CUDA versions 12.4 for the manual installation.

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   git clone https://github.com/lum3on/comfyui_HiDream-Sampler ComfyUI/custom_nodes/comfui_HiDream-Sampler

2. Install requirements
    ```bash
    pip install -r requirements.txt

3. Restart ComfyUI.

## Usage
- Add the HiDreamSampler node to your workflow.
- Configure inputs:
    model_type: Choose full, dev, or fast.
    prompt: Enter your text prompt (e.g., "A photo of an astronaut riding a horse on the moon").
    resolution: Select from available options (e.g., "1024 × 1024 (Square)").
    seed: Set a random seed.
    override_steps and override_cfg: Optionally override default steps and guidance scale.
- Connect the output to a PreviewImage or SaveImage node.

## Requirements
- ComfyUI
- CUDA-enabled GPU (for model inference)

## Notes
Models are cached after the first load to improve performance and use 4-bit quantization.
Ensure you have sufficient VRAM (e.g., 12GB+ recommended for full mode).
