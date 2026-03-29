#!/usr/bin/env python3
"""
GPU Memory Guard - CLI utility to check VRAM before loading AI models.

Prevents out-of-memory crashes by checking available GPU VRAM and
estimating whether a model will fit with a safety buffer.
"""

import json
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple


@dataclass
class GPUInfo:
    """GPU memory information."""
    device_id: int
    name: str
    total_memory_gb: float
    used_memory_gb: float
    available_memory_gb: float
    utilization_percent: float


def get_nvidia_smi_info() -> Optional[List[GPUInfo]]:
    """Query GPU info using nvidia-smi."""
    import subprocess

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,nounits,noheader",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            try:
                device_id = int(parts[0])
                name = parts[1]
                total_mb = float(parts[2])
                used_mb = float(parts[3])
                free_mb = float(parts[4])
                util = float(parts[5])

                gpu = GPUInfo(
                    device_id=device_id,
                    name=name,
                    total_memory_gb=total_mb / 1024.0,
                    used_memory_gb=used_mb / 1024.0,
                    available_memory_gb=free_mb / 1024.0,
                    utilization_percent=util,
                )
                gpus.append(gpu)
            except (ValueError, IndexError):
                continue

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def get_pynvml_info() -> Optional[List[GPUInfo]]:
    """Query GPU info using pynvml."""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpu = GPUInfo(
                device_id=i,
                name=name,
                total_memory_gb=mem_info.total / (1024**3),
                used_memory_gb=mem_info.used / (1024**3),
                available_memory_gb=mem_info.free / (1024**3),
                utilization_percent=util.gpu,
            )
            gpus.append(gpu)

        pynvml.nvmlShutdown()
        return gpus

    except (ImportError, Exception):
        return None


def get_gpu_info() -> Optional[List[GPUInfo]]:
    """Get GPU information. Try pynvml first, fall back to nvidia-smi."""
    info = get_pynvml_info()
    if info is not None:
        return info

    info = get_nvidia_smi_info()
    if info is not None:
        return info

    return None


def check_vram(model_size_gb: float, buffer_gb: float = 0.5) -> Tuple[bool, str]:
    """
    Check if sufficient VRAM is available for a model.

    Args:
        model_size_gb: Model size in GB
        buffer_gb: Safety buffer in GB (default 0.5)

    Returns:
        (can_fit, message) tuple
    """
    gpu_info = get_gpu_info()

    if gpu_info is None:
        msg = "Unable to detect GPU. Ensure nvidia-smi or pynvml is available."
        return (False, msg)

    if not gpu_info:
        return (False, "No GPUs detected.")

    total_available = sum(gpu.available_memory_gb for gpu in gpu_info)
    required = model_size_gb + buffer_gb

    can_fit = total_available >= required

    message = f"Total available: {total_available:.2f}GB, required: {required:.2f}GB"

    return (can_fit, message)


def can_load_model(model_size_gb: float, buffer_gb: float = 0.5) -> bool:
    """
    Check if a model can be loaded without OOM.

    Args:
        model_size_gb: Model size in GB
        buffer_gb: Safety buffer in GB

    Returns:
        True if the model fits, False otherwise
    """
    fits, _ = check_vram(model_size_gb, buffer_gb)
    return fits


def format_human_output(gpu_info, model_size_gb=None, buffer_gb=0.5):
    """Format GPU info for human-readable output."""
    lines = []
    lines.append("GPU Memory Status")
    lines.append("=" * 60)

    total_available = 0
    for gpu in gpu_info:
        lines.append(f"\nGPU {gpu.device_id}: {gpu.name}")
        lines.append(f"  Total:     {gpu.total_memory_gb:>7.2f}GB")
        lines.append(f"  Used:      {gpu.used_memory_gb:>7.2f}GB")
        lines.append(f"  Available: {gpu.available_memory_gb:>7.2f}GB")
        lines.append(f"  Util:      {gpu.utilization_percent:>7.1f}%")
        total_available += gpu.available_memory_gb

    lines.append("\n" + "-" * 60)
    lines.append(f"Total available across all GPUs: {total_available:.2f}GB")

    if model_size_gb is not None:
        required = model_size_gb + buffer_gb
        lines.append(f"\nModel size:     {model_size_gb:.2f}GB")
        lines.append(f"Safety buffer:  {buffer_gb:.2f}GB")
        lines.append(f"Total required: {required:.2f}GB")
        lines.append("-" * 60)

        if total_available >= required:
            margin = total_available - required
            lines.append(f"\u2713 Model WILL fit ({margin:.2f}GB margin)")
        else:
            deficit = required - total_available
            lines.append(f"\u2717 Model will NOT fit (need {deficit:.2f}GB more)")

    return "\n".join(lines)


def format_json_output(gpu_info, model_size_gb=None, buffer_gb=0.5):
    """Format output as JSON."""
    total_available = sum(gpu.available_memory_gb for gpu in gpu_info)

    output = {
        "gpus": [asdict(gpu) for gpu in gpu_info],
        "total_available_gb": total_available,
    }

    if model_size_gb is not None:
        required = model_size_gb + buffer_gb
        output.update({
            "model_size_gb": model_size_gb,
            "buffer_gb": buffer_gb,
            "total_required_gb": required,
            "can_fit": total_available >= required,
        })

    return json.dumps(output, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPU Memory Guard - Check VRAM before loading AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current GPU status
  gpu_guard.py

  # Check if a 18GB model fits with 2GB buffer
  gpu_guard.py --model-size 18 --buffer 2

  # JSON output for scripting
  gpu_guard.py --model-size 13 --json

  # Minimal output for shell scripts
  gpu_guard.py --model-size 7 --quiet
        """,
    )

    parser.add_argument(
        "--model-size",
        type=float,
        help="Model size in GB to check",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.5,
        help="Safety buffer in GB (default: 0.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (exit code only)",
    )

    args = parser.parse_args()

    gpu_info = get_gpu_info()

    if gpu_info is None:
        if not args.quiet:
            print("ERROR: Unable to detect GPU.", file=sys.stderr)
            print(
                "Ensure nvidia-smi is installed or pynvml is available.",
                file=sys.stderr,
            )
        sys.exit(2)

    if not gpu_info:
        if not args.quiet:
            print("ERROR: No GPUs detected.", file=sys.stderr)
        sys.exit(2)

    if args.quiet and args.model_size is not None:
        fits, _ = check_vram(args.model_size, args.buffer)
        sys.exit(0 if fits else 1)

    if args.json:
        output = format_json_output(gpu_info, args.model_size, args.buffer)
        print(output)
    else:
        output = format_human_output(gpu_info, args.model_size, args.buffer)
        print(output)

    if args.model_size is not None:
        fits, _ = check_vram(args.model_size, args.buffer)
        sys.exit(0 if fits else 1)

    sys.exit(0)


if __name__ == "__main__":
    main()
