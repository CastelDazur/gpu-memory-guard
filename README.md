# GPU Memory Guard

A simple CLI utility to check available GPU VRAM before loading AI models. Prevents out-of-memory crashes by estimating whether a model will fit with a safety buffer.

## Installation

### Via pip

```bash
pip install gpu-memory-guard
```

With optional pynvml support:

```bash
pip install gpu-memory-guard[pynvml]
```

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

# Check if model fits
fits, message = check_vram(model_size_gb=18, buffer_gb=2)
print(message)
if fits:
    print("Safe to load")
else:
    print("Will OOM")

# Simple boolean check
if can_load_model(model_size_gb=13):
    load_model()
```

## Integration Examples

### Ollama

Before running a model, check VRAM:

```bash
# examples/ollama_guard.sh
./ollama_guard.sh mistral 7
```

### vLLM

Load models with automatic VRAM checking:

```bash
# examples/vllm_guard.py
python vllm_guard.py --model mistralai/Mistral-7B --max-tokens 4096
```

### llama.cpp

```bash
# Check before loading
gpu-guard --model-size 8 && ./main -m model.gguf -n 256
```

## Exit Codes

- `0`: Success (model fits or no check performed)
- `1`: Model does not fit
- `2`: GPU detection failed or no GPUs found

## Features

- Works with multiple GPUs
- Automatic fallback from pynvml to nvidia-smi
- JSON output for scripting
- Human-readable default output
- Safety buffer configuration
- No external dependencies (optional pynvml)

## How It Works

1. Queries GPU memory using pynvml or nvidia-smi
2. Sums available memory across all GPUs
3. Checks if (model_size + buffer) <= available_memory
4. Returns result or exits with appropriate code

## Troubleshooting

### "Unable to detect GPU"

Ensure nvidia-smi is installed and accessible:

```bash
which nvidia-smi
```

Or install pynvml:

```bash
pip install pynvml
```

### Inaccurate memory readings

- Close other applications using GPU
- Run nvidia-smi directly to verify
- Check GPU driver is up-to-date

### Multiple GPUs not detected

Verify with:

```bash
nvidia-smi -L
```

## License

MIT
