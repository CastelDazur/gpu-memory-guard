# GPU Memory Guard

[![test](https://github.com/CastelDazur/gpu-memory-guard/actions/workflows/test.yml/badge.svg)](https://github.com/CastelDazur/gpu-memory-guard/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/gpu-memory-guard.svg)](https://pypi.org/project/gpu-memory-guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A small CLI that checks free NVIDIA GPU memory before you load a model. It compares free VRAM with the model size plus a safety buffer and tells you, in a terminal message or an exit code, whether the model is likely to fit.

It is a pre-load estimate, not a guarantee. See [Limits](#limits).

## Why?

Loading a model that does not fit can end in an out-of-memory error, a hung GPU or, on some desktop setups, a frozen session. A one-second check first lets a script pick a smaller model or wait until memory is free.

| Without a check | With gpu-memory-guard |
|---|---|
| Start loading a 70B model on a 24 GB card | Check free VRAM **before** loading |
| Find out from an OOM error or a hang | Get a clear message or exit code |
| Free memory and try again | Pick a smaller model or quantization up front |

## Quick start

```bash
pip install gpu-memory-guard
```

```bash
# Current GPU status
gpu-guard

# Is there room for an 18 GB model with a 2 GB safety buffer?
gpu-guard --model-size 18 --buffer 2
```

**Real output** (RTX 5090, nothing else loaded):

```
GPU Memory Status
============================================================

GPU 0: NVIDIA GeForce RTX 5090
  Total:       31.84GB
  Used:         0.99GB
  Available:   30.44GB
  Util:          0.0%

------------------------------------------------------------
Total available across all GPUs: 30.44GB

Model size:     18.00GB
Safety buffer:  2.00GB
Total required: 20.00GB
------------------------------------------------------------
✓ Estimated to fit (10.44GB margin)
```

## Documentation

- [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) - Sizing reference for GPUs, models, and quantizations (Q4_K_M, Q5_K_M, Q8_0, FP16) with KV cache tables and the mmproj trap for vision-language models.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Field guide to the five CUDA OOM errors you will actually see, with a diagnostic checklist and notes on vLLM, llama.cpp, and Ollama quirks.

## Installation

```bash
pip install gpu-memory-guard            # uses nvidia-smi
pip install "gpu-memory-guard[pynvml]"  # also installs pynvml
```

From source:

```bash
git clone https://github.com/CastelDazur/gpu-memory-guard.git
cd gpu-memory-guard
pip install -e .
```

### Requirements

- Python 3.8+
- An NVIDIA GPU and driver, with either `nvidia-smi` on `PATH` or the `pynvml` package. `pynvml` is tried first, `nvidia-smi` second.

## Usage

### CLI

```bash
# GPU status only
gpu-guard

# Check a model size in GB (default safety buffer: 0.5 GB)
gpu-guard --model-size 13

# Custom safety buffer
gpu-guard --model-size 18 --buffer 2

# JSON for scripts
gpu-guard --model-size 13 --json

# Exit code only
gpu-guard --model-size 7 --quiet
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Estimated to fit, or status shown without `--model-size` |
| `1` | Estimated NOT to fit |
| `2` | No GPU detected (neither `pynvml` nor `nvidia-smi` worked) |

`--json` prints the per-GPU numbers plus `model_size_gb`, `buffer_gb`, `total_required_gb` and `can_fit` when `--model-size` is given.

### As a Python library

```python
from gpu_guard import can_load_model, check_vram, get_gpu_info

# Per-GPU numbers (None if no GPU could be queried)
for gpu in get_gpu_info() or []:
    print(f"GPU {gpu.device_id}: {gpu.available_memory_gb:.2f} GB available")

# True / False
if can_load_model(model_size_gb=13.0, buffer_gb=2.0):
    print("Estimated to fit")

# (bool, message) with the numbers behind the decision
fits, message = check_vram(model_size_gb=13.0, buffer_gb=2.0)
print(fits, message)  # e.g. True Total available: 30.44GB, required: 15.00GB
```

### Scripting example

```bash
# Pre-check before launching inference
if gpu-guard --model-size 13 --quiet; then
    python run_inference.py --model llama-13b
else
    echo "Probably not enough VRAM, switching to 7B model"
    python run_inference.py --model llama-7b
fi
```

## Limits

- **Free VRAM now is not peak usage later.** The check compares a number you give with memory that is free at this moment. Context length and KV cache, the runtime's own overhead, activation memory and other processes that start afterwards all add to real usage. Enter the size you expect at peak, not just the weights file, and keep a buffer. [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) has sizing tables.
- **Multiple GPUs are summed.** The total is free memory across all cards. A model that is not split across GPUs needs the space on one card; the CLI prints a note when more than one GPU is present.
- **NVIDIA only.** AMD and Apple GPUs are not detected.
- **It does not stop anything.** The tool reports; it does not reserve memory or prevent another process from taking it between the check and the load.

## Common model sizes (approximate VRAM)

| Model | FP16 | Q4 (GGUF) |
|---|---|---|
| 7B params | ~14 GB | ~4 GB |
| 13B params | ~26 GB | ~7 GB |
| 33B params | ~66 GB | ~18 GB |
| 70B params | ~140 GB | ~35 GB |

Weights only; add context/KV cache and runtime overhead.

## Roadmap

- [ ] AMD ROCm support
- [ ] Memory estimation by model architecture
- [ ] Multi-GPU split recommendations
- [ ] Integration with Ollama and vLLM

## Contributing

PRs welcome. If you want to add AMD ROCm support or model-specific memory estimation, open an issue first so we can discuss the approach.

## License

MIT
