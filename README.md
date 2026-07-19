# PixelPilot

Attention-based click grounding for macOS GUI automation.

Uses Qwen3-VL-2B-Instruct cross-attention maps to locate UI elements from natural language descriptions, then clicks them via pyautogui.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
python cli.py --target "the Submit button" --dry-run
```

## Documentation

- [Commands](docs/commands.md) — all available commands for setup, grounding, benchmarks, and tests

## Project Structure

```
grounding/            # attention-based grounding (active)
agent/                # legacy agent loop (llama-cpp)
perception/           # screen parsing providers (ax, moondream, omniparser)
benchmarks/           # tuning & evaluation scripts
tests/                # pytest unit & integration tests
scripts/              # tooling (model download)
refinement/           # refinement CNN training pipeline
cli.py                # main entry point
models/               # model weights (.gitignored)
```
