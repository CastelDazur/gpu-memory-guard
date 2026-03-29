# GPU Memory Guard


A simple CLI utility to check available GPU VRAM before loading AI models. Prevents out-of-memory crashes by estimating whether a model will fit with a safety buffer.


## Installation


### From source


```bash
git clone https://github.com/CastelDazur/gpu-memory-guard.git
cd gpu-memory-guard
pip install -e .
```


## Requirements


- Python 3.8+
- NVIDIA GPU with nvidia-smi installed, OR
- pynvml Python package


## Usage


### CLI


Check current GPU status:


```bash
gpu-guard
```


Check if an 18GB model fits with 2GB buffer:


```bash
gpu-guard --model-size 18 --buffer 2
```


JSON output for scripting:


```bash
gpu-guard --model-size 13 --json
```


Minimal output (exit code only):


```bash
gpu-guard --model-size 7 --quiet
```


### As a Python Library


```python
from gpu_guard import check_vram, can_load_model, get_gpu_info


# Check current VRAM
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu.device_id}: {gpu.available_memory_gb:.2f}GB available")
