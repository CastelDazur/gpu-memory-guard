# Troubleshooting GPU OOM Errors

A field guide to the CUDA out-of-memory errors you will actually see when running local models, and how to diagnose them faster.

This is the document I wish had existed when I started running GGUF models on a workstation. The [README](README.md) covers `gpu-guard` basics; this file covers the surrounding traps that cause 90% of the "but it should fit" incidents.

## Table of contents

- [The five errors you will see](#the-five-errors-you-will-see)
- [The mmproj trap (vision models)](#the-mmproj-trap-vision-models)
- [The KV cache trap (long context)](#the-kv-cache-trap-long-context)
- [The auto-fit trap (llama.cpp)](#the-auto-fit-trap-llamacpp)
- [The "free VRAM" lie](#the-free-vram-lie)
- [The driver fragmentation trap](#the-driver-fragmentation-trap)
- [The parallel request trap (vLLM, TGI)](#the-parallel-request-trap-vllm-tgi)
- [Diagnostic checklist](#diagnostic-checklist)

## The five errors you will see

These are the actual strings you will see in logs. Each maps to a distinct root cause.

### 1. `CUDA error: out of memory` (during load)

The model weights themselves do not fit. Either the file on disk is larger than your available VRAM minus overhead, or another process is already using VRAM.

Fix: run `gpu-guard` before loading. If it reports the model should fit but you still crash, jump to [the free VRAM lie](#the-free-vram-lie).

### 2. `CUDA error: out of memory` (during first inference)

Weights loaded fine, but the first forward pass tried to allocate the KV cache for full context length and ran out. The model is technically loaded, but it cannot run any prompt.

Fix: cap context length. For llama.cpp: `-c 4096`. For Ollama: `num_ctx: 4096` in the model's modelfile or request parameters. For vLLM: `--max-model-len 4096`.

### 3. `CUDA error: out of memory` (during generation, variable prompt length)

You had headroom at 4K context but ran a 16K prompt. The KV cache grew past what the card can hold.

Fix: either cap context lower, or upgrade to a card class where the full context fits. See the KV cache table in [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md).

### 4. `allocation failed: out of memory` (vLLM)

vLLM preallocates a pool for the KV cache based on `gpu_memory_utilization`. If your model barely fits and the pool is oversized, load succeeds but the first request fails.

Fix: lower `--gpu-memory-utilization` from the default 0.90 down to 0.80 or 0.85, and cap `--max-num-seqs` to reduce how many parallel requests the pool has to plan for.

### 5. `cudaMalloc failed: 2` with no other context

This is the least informative error and the most common. It means the driver refused a VRAM allocation. Root cause could be any of the traps below. Do not guess; run through the diagnostic checklist at the bottom.

## The mmproj trap (vision models)

LLaVA, Qwen-VL, MiniCPM-V, Gemma 2 vision, and similar multimodal models ship the vision tower as a separate file with `mmproj` in the name. **It is loaded into VRAM in addition to the language weights.**

Symptoms:
- "Model should fit" according to file size of the main GGUF
- Load succeeds, but VRAM usage is 700 MB to 2 GB higher than expected
- Crashes on the first image input

What to check:
- Did you download the mmproj file? If not, vision features silently break
- Does the mmproj file size plus the main model fit with a buffer?
- Did llama.cpp or the wrapper account for both files in its logging?

Fix: when you size a VLM, size it as `base + mmproj + kv_cache + 1 GB overhead`, not just the base model. `gpu-guard --model-size $TOTAL` with the combined total.

## The KV cache trap (long context)

The KV cache scales linearly with sequence length and with batch size. For Llama 3.3 70B at 4K context the KV cache is about 1.25 GB. At 128K context it is about 40 GB. On a 48 GB card, running 70B Q4_K_M with 128K context means weights at 40 GB plus KV at 40 GB equals a crash that the weight-only math would have said fits.

Symptoms:
- Load succeeds
- Short prompts work
- Crashes on long prompts, long generations, or when the server is asked to serve multiple concurrent requests

What to check:
- What context length is the runtime actually configured for? Not what you passed in a single request, but the runtime's internal cap
- Are you running a server that pre-allocates the KV cache for maximum context? vLLM does by default; llama.cpp does if you pass a large `-c`
- Is the model a long-context variant with a native 128K or 1M token window? Those advertise the window, but you rarely need it

Fix: cap context to what you actually use. Most production workloads fit in 4K to 16K. A 32K cap is the sweet spot for agentic coding tasks. 128K is for document analysis and retrieval over huge corpora, and you should explicitly plan VRAM for it.

## The auto-fit trap (llama.cpp)

llama.cpp has an `--n-gpu-layers` or `-ngl` flag and an `auto` mode. The auto mode computes how many layers it thinks will fit. It is optimistic because:

- It may not count the mmproj file for VLMs
- It does not always count KV cache at full context
- It does not count CUDA kernel overhead for large batches

Symptoms:
- First run works, subsequent runs crash
- Works at batch size 1, crashes at batch size 4
- Works in llama-cli, crashes in llama-server

Fix: do not rely on auto. Set `-ngl` explicitly. Start with the value auto suggests, subtract 2, and test. If you still crash, subtract 2 more. The fallback layers run on CPU but the model still serves; you pay throughput to buy stability.

## The "free VRAM" lie

`nvidia-smi` reports memory as free that is not actually free for your process. The common causes:

- **Another CUDA process holds VRAM**. Chrome with hardware acceleration, your desktop compositor, or a forgotten Jupyter kernel can hold 1-3 GB
- **Fragmentation from prior loads**. Even after a process exits, the allocator can leave the VRAM in a state where no large contiguous block is available
- **`MIG` partitions** on datacenter cards. An H100 in MIG mode shows the full 80 GB, but only a partition is allocatable to your process
- **Unified memory overflow**. If a prior process used managed memory, the driver can hold onto mappings

Symptoms:
- `nvidia-smi` shows 20 GB free, but `cudaMalloc` of 15 GB fails
- Works after reboot, fails after the machine has been up for a while

What to check:
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` to see what processes actually hold VRAM
- `nvidia-smi --query-gpu=memory.free,memory.reserved --format=csv` to see if the driver itself is reserving memory
- Kill stale Python and Jupyter processes before loading

Fix: `gpu-guard` reads available memory the same way the driver reports it, so if gpu-guard says 20 GB free and the load still fails, you have fragmentation or another process. Run `fuser -k /dev/nvidia*` to kill anything holding the device, then retry.

## The driver fragmentation trap

After many load/unload cycles, VRAM can fragment such that no contiguous block large enough for the model weights is available, even though the total free bytes are plenty.

Symptoms:
- Cold boot: load works
- After 5-10 reloads in a session: load fails with OOM despite "enough free" VRAM
- `nvidia-smi` looks fine

Fix:
- Restart the runtime process between reloads (kill the Python process, do not just reassign the model variable)
- On Linux: `nvidia-smi --gpu-reset` as root (do not do this on a machine serving other workloads)
- Last resort: `sudo rmmod nvidia_uvm nvidia_modeset nvidia; sudo modprobe nvidia` to reset the driver without a reboot

## The parallel request trap (vLLM, TGI)

vLLM and Text Generation Inference preallocate KV cache pools sized for the maximum number of parallel requests they are told to handle. If you set `--max-num-seqs 256` on a card that can only hold 16, the server will report "loaded" and then fail on the second concurrent request.

Symptoms:
- `/v1/models` endpoint responds
- Single-request inference works
- Concurrent requests or sustained load cause allocation failures or 500 responses

What to check:
- What is your `--max-num-seqs` or equivalent setting?
- What is your `--max-model-len`?
- Does `gpu_memory_utilization × total_vram - weights - kv_for_max_seqs` go negative?

Fix: start with `--max-num-seqs 4` on consumer cards, `--max-num-seqs 16` on 48 GB cards, `--max-num-seqs 64` on 80 GB cards. Scale up only when you have measured headroom.

## Diagnostic checklist

When the model crashes on load and you do not know why, run this list in order. It covers 95% of the real causes.

```
1. gpu-guard                                  # what does the tool report?
2. nvidia-smi                                 # are there other processes holding VRAM?
3. nvidia-smi --query-gpu=memory.free,memory.reserved --format=csv
4. ls -lah $MODEL_FILE                        # is the file what you think it is?
5. file $MODEL_FILE                           # is it the right format?
6. gpu-guard --model-size $(file_size_gb)     # does it fit with a 2 GB buffer?
7. Check for mmproj file                      # VLM? add its size
8. Check configured context length            # cap to 4K for a baseline test
9. Check --max-num-seqs or equivalent         # scale to 1 for a baseline test
10. Restart the runtime process cleanly       # kill fragmentation
11. If still failing: reboot                  # the boring answer that works
```

If you have gone through this list and you still cannot explain the failure, open an issue on the gpu-memory-guard repo with:
- GPU model and driver version (`nvidia-smi --query-gpu=name,driver_version --format=csv`)
- Runtime (llama.cpp version, Ollama version, vLLM version)
- Model file name and size
- The exact error message
- Output of `gpu-guard --json`

That will usually be enough to diagnose remotely.

## See also

- [README.md](README.md) for basic CLI usage
- [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) for sizing reference tables
- [examples/](examples/) for runtime-specific wrappers
