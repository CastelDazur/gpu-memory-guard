# Model Compatibility Reference

A practical reference for deciding which open-weight model fits on which GPU, at which quantization, with how much context.

This file is intentionally math-first: every number comes from the standard quantization formulas and widely reported model parameter counts, so you can reproduce it yourself. If you need the runtime check, use `gpu-guard` (see [README](README.md)).

## TL;DR cheat sheet

This table shows the **smallest GPU class** (by VRAM) that can load each model at each common quantization with a 2 GB safety buffer and 4K context. Longer context or parallel requests will push requirements higher; see the [KV cache section](#kv-cache-the-part-that-surprises-people).

| Model | Params | Q4_K_M | Q5_K_M | Q8_0 | FP16 |
|---|---|---|---|---|---|
| Phi-4 mini | 3.8B | 8 GB | 8 GB | 12 GB | 12 GB |
| Llama 3.1 8B | 8B | 8 GB | 12 GB | 12 GB | 24 GB |
| Mistral 7B | 7.2B | 8 GB | 8 GB | 12 GB | 16 GB |
| Gemma 2 9B | 9B | 12 GB | 12 GB | 16 GB | 24 GB |
| Qwen 2.5 14B | 14B | 12 GB | 16 GB | 24 GB | 32 GB |
| Phi-4 | 14B | 12 GB | 16 GB | 24 GB | 32 GB |
| Gemma 2 27B | 27B | 24 GB | 24 GB | 32 GB | 80 GB |
| Qwen 2.5 32B | 32B | 24 GB | 32 GB | 48 GB | 80 GB |
| Mixtral 8x7B | 46.7B | 32 GB | 48 GB | 80 GB | 2x48 GB |
| Llama 3.3 70B | 70B | 48 GB | 80 GB | 80 GB | 2x80 GB |
| Qwen 2.5 72B | 72B | 48 GB | 80 GB | 80 GB | 2x80 GB |
| Llama 3.1 405B | 405B | 2x80 GB | 4x80 GB | 6x80 GB | 8x80 GB |

Read as: "the first column where the model still fits with a 2 GB buffer."

## The math you can do in your head

For dense transformers (Llama, Mistral, Qwen, Gemma, Phi), memory for the weights alone is:

```
weight_bytes ≈ params × bits_per_param / 8
```

Typical `bits_per_param` for common GGUF quantizations:

| Quant | Effective bits | bytes/param |
|---|---|---|
| FP16 / BF16 | 16 | 2.00 |
| Q8_0 | 8.5 | 1.06 |
| Q6_K | 6.56 | 0.82 |
| Q5_K_M | 5.50 | 0.69 |
| Q5_K_S | 5.25 | 0.66 |
| Q4_K_M | 4.58 | 0.57 |
| Q4_K_S | 4.25 | 0.53 |
| Q3_K_M | 3.52 | 0.44 |
| Q2_K | 2.73 | 0.34 |

So for Llama 3.3 70B at Q4_K_M:

```
70 × 10^9 × 0.57 bytes/param ≈ 40 GB
```

Add roughly 1 GB of CUDA context overhead for llama.cpp and friends, and you get the ~41 GB figure that matches what you see in `nvidia-smi` at idle right after the model loads. A 48 GB card (5090 is 32 GB, but an RTX 6000 Ada or L40 is 48 GB) fits comfortably. A 32 GB 5090 does not.

For Mixtral and other mixture-of-experts models, **you still need memory for every expert**, not just the active ones. Mixtral 8x7B is about 46.7B parameters on disk even though only ~13B are active per token. Multiply by the quant ratio the same way.

## Popular GPUs and what they actually do

| GPU | VRAM | Practical budget with 2 GB buffer | Practical model ceiling |
|---|---|---|---|
| RTX 4060 | 8 GB | 6 GB | 7B to 8B at Q4_K_M, short context |
| RTX 4060 Ti 16 GB | 16 GB | 14 GB | 14B at Q4_K_M; 8B at Q8_0 |
| RTX 4070 | 12 GB | 10 GB | 9B at Q4_K_M; 7B at Q5_K_M |
| RTX 4070 Ti Super | 16 GB | 14 GB | 14B at Q4_K_M; 9B at Q8_0 |
| RTX 4080 Super | 16 GB | 14 GB | 14B at Q4_K_M; 9B at Q8_0 |
| RTX 4090 | 24 GB | 22 GB | 27B at Q4_K_M; 14B at Q8_0 |
| RTX 5080 | 16 GB | 14 GB | same class as 4080 Super for this purpose |
| RTX 5090 | 32 GB | 30 GB | 32B at Q4_K_M; 27B at Q5_K_M; 14B at Q8_0 |
| RTX 6000 Ada | 48 GB | 46 GB | 70B at Q4_K_M; 32B at Q8_0 |
| L40 / L40S | 48 GB | 46 GB | 70B at Q4_K_M; 32B at Q8_0 |
| A100 40 GB | 40 GB | 38 GB | 70B at Q3_K_M; 32B at Q5_K_M |
| A100 80 GB | 80 GB | 78 GB | 70B at Q8_0; 72B at Q5_K_M |
| H100 80 GB | 80 GB | 78 GB | 70B at Q8_0; 72B at Q5_K_M |
| 2x H100 | 160 GB | 156 GB | 70B at FP16; 405B at Q2_K |

"Practical ceiling" assumes 4K context and no concurrent requests. If you run a batch serving setup, cut the ceiling by 20 to 40 percent depending on `max_concurrent_requests` and context length.

## KV cache: the part that surprises people

The weights are only half the story. Every token in your prompt plus every token you generate needs memory in the KV cache. For a transformer with `L` layers, `H` KV heads, `D` head dim, at FP16:

```
kv_cache_bytes ≈ 2 × L × H × D × sequence_length × 2 bytes
```

Or more practically, for common models with a 4K context window:

| Model | KV cache @ 4K | KV cache @ 32K | KV cache @ 128K |
|---|---|---|---|
| Llama 3.1 8B | ~0.5 GB | ~4 GB | ~16 GB |
| Llama 3.3 70B | ~1.25 GB | ~10 GB | ~40 GB |
| Mistral 7B | ~0.5 GB | ~4 GB | n/a (8K native) |
| Qwen 2.5 32B | ~1 GB | ~8 GB | ~32 GB |

If you try to run Llama 3.3 70B at Q4_K_M on a 48 GB card with 128K context, the weights fit at ~40 GB, but the KV cache adds another ~40 GB, and you get a crash that looks nothing like "the model is too big." The real cause is the sum.

llama.cpp, Ollama, and vLLM all let you cap context length. If you do not cap it, they will happily allocate for the full trained context, even when your request is 200 tokens.

## Quantizations you should actually consider

**Q4_K_M** is the default for a reason: about 4.58 bits per weight, minimal perplexity loss versus Q5_K_M in most benchmarks, and it unlocks a full class of bigger models on the same card. Start here.

**Q5_K_M** is worth it when you can afford the extra VRAM and you are running a reasoning-heavy workload (code generation, math, multi-step agents) where the last few percent of capability show up in real output quality. On a 4090 you can run Qwen 2.5 14B at Q5_K_M instead of Q4_K_M at almost no cost.

**Q8_0** is effectively lossless relative to FP16 for inference, and it costs half the memory. If you have a 48 GB or 80 GB card, there is rarely a reason to run FP16 for serving.

**FP16 / BF16** makes sense for fine-tuning, for models where the quantization hurts (very small models, some reasoning models, certain vision-language models where the projector is sensitive), and for research reproducibility. For everyday serving, it is wasteful.

**Q2_K and Q3_K** are emergency quants. Use them when you need to run a model that is one class above your GPU and you will tolerate the quality hit. Q2_K on a 70B model is still better than Q4_K_M on a 32B model for some tasks, but not for most.

## Vision-language models and the mmproj trap

Multimodal models like LLaVA, Qwen-VL, and Gemma 2 vision variants ship the vision tower as a separate file, typically named something with `mmproj` in it. **That file is loaded into VRAM in addition to the base model.**

For Qwen 2.5 VL 7B:
- Base weights at Q4_K_M: ~4.5 GB
- mmproj at Q8: ~0.7 GB
- KV cache at 4K: ~0.5 GB
- Total: ~5.7 GB, not 4.5 GB

Auto-fit logic in llama.cpp does not always include the mmproj file in its calculation. This is a recurring source of "but it should fit" bug reports. If you are checking a VLM, size it with the mmproj added.

## Multi-GPU splits

If your model does not fit on one card, llama.cpp and vLLM will split it across GPUs. The split is not free: there is a small overhead per layer crossing a PCIe boundary, and it is much larger if you do not have NVLink. As a rule of thumb:

- **Two identical cards (e.g., 2x 4090)**: efficient, expect ~90% of single-card speed per token at equivalent batch size.
- **Two different cards (e.g., 4090 + 3090)**: still works; llama.cpp supports manual layer offloading. Slower than identical cards.
- **CPU offload**: the model runs, but you will see 5x to 20x slowdowns depending on how many layers end up on CPU. Useful for prototyping, not for serving.

## Sources and method

Parameter counts come from the official model cards. Quantization bit-per-weight figures come from llama.cpp's `k-quant` documentation and are averages; actual values vary slightly per layer because the k-quants mix block sizes.

These numbers are meant to be accurate to within about 5%. For a precise figure on a specific build, run:

```bash
gpu-guard --model-size $(stat -c %s model.gguf | awk '{print $1/1024/1024/1024}') --buffer 2
```

or just let the guard read the file header directly when that support lands.

## Contributing

If you have verified numbers for a model or GPU that is not on this table, open a PR with:
- Model name and parameter count
- Quantization file URL (Hugging Face preferred)
- GPU model and driver version
- `nvidia-smi` output after the model loads and is warm
- Context length used

Hardware reports are welcome. We would rather have real measurements than formulas where we can get them.
