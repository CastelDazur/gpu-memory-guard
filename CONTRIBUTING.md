# Contributing to GPU Memory Guard

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/CastelDazur/gpu-memory-guard.git
cd gpu-memory-guard
pip install -e ".[dev]"
```

## Running Tests

Tests use mocked GPU data and run without a physical GPU:

```bash
pytest tests/ -v
```

## What We're Looking For

- **Bug fixes** with a test that reproduces the issue
- **New GPU backends** (AMD ROCm, Intel Arc, Apple Metal)
- **GGUF size estimation** from model metadata
- **Multi-GPU split logic** for tensor parallelism
- **Documentation** improvements and examples

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Make sure all tests pass
4. Update README.md if you changed the public API
5. Submit a PR with a clear description

## Code Style

- Python 3.8+ compatible
- Type hints on all public functions
- Docstrings for classes and public methods
- No external dependencies in the core module

## Reporting Bugs

Open an issue with:
- Your GPU model and driver version
- Python version
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
