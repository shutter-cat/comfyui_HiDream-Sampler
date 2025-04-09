![image](https://github.com/user-attachments/assets/3d4e9bee-772b-4c57-84cb-b5a6da30efd5)

# HiDreamSampler for ComfyUI

A custom ComfyUI node for generating images using the HiDream AI model.

## Features
- Supports `full`, `dev`, and `fast` model types.
- Configurable resolution and inference steps.
- Uses 4-bit quantization for lower memory usage.

## Installation
Please make sure you have installed Flash Attention. We recommend CUDA versions 12.4 for the manual installation.

- Get Flash-Attention 2 wheel from [HuggingFace](https://huggingface.co/lldacing/flash-attention-windows-wheel/blob/main/flash_attn-2.7.4%2Bcu126torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl) (Python 3.12, PyTorch 2.6.0, cuda 12.6, other available there too)
- Install it in ComfyUI (.\python_embeded\python.exe -s -m pip install file.whl for portable version)
- Install accelerate .\python_embeded\python.exe -s -m pip install accelerate>=0.26.0
- Install this node with ComfyManager (or manually, don't forget to call python_embeded etc for portable version)
- add --use-flash-attention in "run_nvidia_gpu.bat"
- Use the "HiDream Sampler" node once to download the model

(If you don't want to install random wheel, you can take it from here (it should create a release once it finish, which should take ~2 hours on GitHub CI))

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
Models are cached after the first load to improve performance and use 4-bit quantization models from https://github.com/hykilpikonna/HiDream-I1-nf4.
Ensure you have sufficient VRAM (e.g., 16GB+ recommended for full mode).
