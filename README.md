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
   git clone https://github.com/lum3on/your_repo_name.git ComfyUI/custom_nodes/your_repo_name3

2. Install requirements
    ```bash
    pip install -r requirements.txt