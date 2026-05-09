#!/usr/bin/env python
# perception/__main__.py
"""
Run as:
    python -m perception                        # parse with default provider (ax)
    python -m perception --provider omniparser  # use specific provider
    python -m perception --list                 # list registered providers
    python -m perception --benchmark            # benchmark all providers
    python -m perception --benchmark ax moondream --runs 5
"""

import os

os.environ["HF_HOME"] = os.path.abspath("./models")


import argparse
import sys

import pyautogui
from PIL import Image

from perception import (
    parse_screen,
    elements_to_text,
    benchmark,
    list_providers,
    set_provider,
)


def main():
    parser = argparse.ArgumentParser(prog="python -m perception")
    parser.add_argument("--provider", "-p", default=None, help="Provider to use")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all registered providers"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        nargs="*",
        help="Benchmark providers (optionally name specific ones)",
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3, help="Runs per provider in benchmark mode"
    )
    parser.add_argument(
        "--image", default=None, help="Path to image file instead of live screenshot"
    )
    args = parser.parse_args()

    # ── list ──────────────────────────────────────────────────────────────────
    if args.list:
        print("\nRegistered perception providers:\n")
        for p in list_providers():
            print(f"  {p['name']:<15}  {p['description']}")
        print()
        return

    # ── load image if provided ────────────────────────────────────────────────
    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"Using image: {args.image}")

    # ── benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark is not None:
        names = args.benchmark if args.benchmark else None
        benchmark(provider_names=names, runs=args.runs, image=image)
        return

    # ── single parse ──────────────────────────────────────────────────────────
    provider = args.provider
    result = parse_screen(image=image, provider=provider)

    print(f"\nProvider : {result['provider']}")
    print(f"Elements : {result['count']}")
    print(f"Time     : {result['parse_time']}s\n")
    print(elements_to_text(result["elements"]))


if __name__ == "__main__":
    main()
