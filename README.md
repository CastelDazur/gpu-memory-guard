# GPU Memory Guard

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/CastelDazur/gpu-memory-guard?style=social)](https://github.com/CastelDazur/gpu-memory-guard/stargazers)

A CLI utility that checks available GPU VRAM before you load AI models. Prevents OOM crashes that force a full system reboot.

## Why?

If you run local inference on consumer GPUs, you know the pain:

| Without gpu-memory-guard | With gpu-memory-guard |
|---|---|
| Load 70B model on 24GB card | Check VRAM **before** loading |
| System freezes, GPU hangs | Get a clear warning in terminal |
| Force reboot, lose unsaved work | Pick a smaller model or free memory |
| Repeat next week | Zero OOM crashes |

One command saves you from constant reboots.

## Quick Start

```bash
git clone https://github.com/CastelDazur/gpu-memory-guard.git
cd gpu-memory-guard
pip install -e .
```

```bash
# Check current GPU status
gpu-guard

# Check if an 18GB model fits with 2GB safety buffer
gpu-guard --model-size 18 --buffer 2
```

**Example output:**

```
GPU 0: NVIDIA GeForce RTX 5090
  Total:     32.00 GB
  Used:       4.12 GB
  Available: 27.88 GB

Model size: 18.00 GB (buffer: 2.00 GB)
Status: OK - model fits with 7.88 GB to spare
```

## Documentation

- [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) - Sizing reference for GPUs, models, and quantizations (Q4_K_M, Q5_K_M, Q8_0, FP16) with KV cache tables and the mmproj trap for vision-language models.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Field guide to the five CUDA OOM errors you will actually see, with a diagnostic checklist and notes on vLLM, llama.cpp, and Ollama quirks.

## Installation

### From source (recommended)

```bash
git clone https://github.com/CastelDazur/gpu-memory-guard.git
cd gpu-memory-guard
pip install -e .
```

### Requirements

- Python 3.8+
- NVIDIA GPU with `nvidia-smi` installed, OR
- `pynvml` Python package (`pip install pynvml`)

## Usage

### CLI

```bash
# Basic VRAM check
gpu-guard

# Check if a model fits (size in GB)
gpu-guard --model-size 13

# Custom safety buffer (default: 1GB)
gpu-guard --model-size 18 --buffer 2

# JSON output for scripting
gpu-guard --model-size 13 --json

# Quiet mode: exit code only (0 = fits, 1 = doesn't)
gpu-guard --model-size 7 --quiet
```

### As a Python library

```python
from gpu_guard import check_vram, can_load_model, get_gpu_info

# Check current VRAM
gpu_info = get_gpu_info()
for gpu in gpu_info:
    print(f"GPU {gpu.device_id}: {gpu.available_memory_gb:.2f}GB available")

# Check if a model fits
result = can_load_model(model_size_gb=13.0, buffer_gb=2.0)
if result.fits:
    print("Safe to load")
else:
    print(f"Need {result.shortage_gb:.2f}GB more VRAM")
```

### Scripting example

```bash
# Pre-check before launching inference
if gpu-guard --model-size 13 --quiet; then
    python run_inference.py --model llama-13b
else
    echo "Not enough VRAM, switching to 7B model"
    python run_inference.py --model llama-7b
fi
```

## Common model sizes (approximate VRAM)

| Model | FP16 | Q4 (GGUF) |
|---|---|---|
| 7B params | ~14 GB | ~4 GB |
| 13B params | ~26 GB | ~7 GB |
| 33B params | ~66 GB | ~18 GB |
| 70B params | ~140 GB | ~35 GB |

## Roadmap

- [ ] AMD ROCm support
- [ ] Memory estimation by model architecture
- [ ] Multi-GPU split recommendations
- [ ] PyPI package (`pip install gpu-memory-guard`)
- [ ] Integration with Ollama and vLLM

## Contributing

PRs welcome. If you want to add AMD ROCm support or model-specific memory estimation, open an issue first so we can discuss the approach.

## License

MIT
