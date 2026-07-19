# Commands

## Setup

```bash
# Clone and enter project
git clone <repo-url>
cd pixel-pilot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (pytest)
pip install pytest

# Download model weights (~4GB)
python scripts/download_models.py
```

## Grounding CLI

```bash
# Click a target element (captures screen automatically)
python cli.py --target "the Submit button"

# Dry run (saves preview image instead of clicking)
python cli.py --target "the red square" --dry-run

# Use a specific image instead of live screenshot
python cli.py --target "the search bar" --image screenshot.png

# Adjust parameters
python cli.py --target "the close button" --temperature 0.05 --top-k-frac 0.05 --retina-scale 2.0
```

## Benchmarks

```bash
# Run full benchmark (generates HTML report in benchmarks/results/)
python -m benchmarks.run

# Sweep token selection, temperature, top-k
python -m benchmarks.tune_params

# Sweep decoder layer selection
python -m benchmarks.tune_layers

# Per-token per-layer attention diagnosis
python -m benchmarks.diagnose
```

## Tests

```bash
# Run unit tests (no model needed)
pytest tests/test_patch_map.py -v

# Run integration tests (requires model downloaded)
pytest tests/test_synthetic_grounding.py -v -s -m slow

# Run all tests
pytest tests/ -v
```

## Refinement Net

```bash
# Step 1: Download ScreenSpot dataset (1,272 samples)
python -m refinement.download_data

# Step 2: Generate attention heatmaps for all samples (~5-6 hours on M3)
# Use caffeinate to prevent sleep during overnight runs
caffeinate -i python -m refinement.generate_heatmaps

# Step 3: Train refinement net
python -m refinement.train

# Step 4: Compare baseline vs refined
python -m refinement.evaluate
```

## Legacy Agent Loop

```bash
# Run the perceive-think-execute agent loop (requires llama-cpp server)
python -m agent.loop "Open Spotlight and search for Calculator"
```
