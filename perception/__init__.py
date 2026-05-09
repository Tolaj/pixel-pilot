# perception/__init__.py
"""
Perception package — modular screen understanding backends.

Quick usage:
    from perception import parse_screen, set_provider

    set_provider("ax")                    # or "omniparser" / "moondream"
    result = parse_screen()

Benchmark:
    from perception import benchmark
    benchmark(["ax", "omniparser", "moondream"], runs=3)

Add a new provider:
    1. Create perception/providers/myprovider.py
    2. Subclass LazyProvider, set name = "myprovider", implement parse()
    3. Decorate with @register
    4. Add `from . import myprovider` in perception/providers/__init__.py
    Done — it shows up in list_providers() and benchmark() automatically.
"""

import os
import time
import pyautogui
from PIL import Image

from .registry import get, list_providers
from . import providers  # noqa: F401 — triggers all @register decorators

# ── active provider (default from env or "ax") ────────────────────────────────
_PROVIDER_NAME = os.environ.get("PERCEPTION_PROVIDER", "ax")
_active_provider = None


def set_provider(name: str, warm_up: bool = False):
    """Switch the active provider at runtime."""
    global _active_provider, _PROVIDER_NAME
    _active_provider = get(name)
    _PROVIDER_NAME = name
    if warm_up:
        print(f"[perception] Warming up '{name}'...")
        _active_provider.warm_up()
        print(f"[perception] '{name}' ready.")


def _get_provider():
    global _active_provider
    if _active_provider is None:
        set_provider(_PROVIDER_NAME)
    return _active_provider


# ── public api ────────────────────────────────────────────────────────────────


def take_screenshot() -> Image.Image:
    return pyautogui.screenshot()


def parse_screen(image: Image.Image = None, provider: str = None) -> dict:
    """
    Capture and parse the current screen.

    Args:
        image:    optional pre-captured PIL image (skips screenshot)
        provider: optional one-shot provider override (doesn't change active provider)

    Returns:
        {
            elements:    list[dict]   — id, label, role, center, bbox, conf, source
            count:       int
            screenshot:  PIL.Image
            parse_time:  float        — seconds
            provider:    str          — which backend was used
        }
    """
    if image is None:
        image = take_screenshot()

    p = get(provider) if provider else _get_provider()

    start = time.time()
    elements = p.parse(image)
    elapsed = round(time.time() - start, 3)

    # re-index so ids are always sequential regardless of what provider returns
    for i, el in enumerate(elements):
        el["id"] = i

    return {
        "elements": elements,
        "count": len(elements),
        "screenshot": image,
        "parse_time": elapsed,
        "provider": p.name,
    }


def elements_to_text(elements: list) -> str:
    lines = [
        f"[{el['id']}] label='{el['label']}' "
        f"role={el.get('role', '?')} "
        f"center={el['center']} "
        f"source={el.get('source', '?')}"
        for el in elements
    ]
    return "\n".join(lines)


def benchmark(
    provider_names: list[str] = None,
    runs: int = 3,
    image: Image.Image = None,
):
    """
    Run each provider N times on the same screenshot and print a comparison table.

    Usage:
        from perception import benchmark
        benchmark(["ax", "omniparser"], runs=3)
    """
    if provider_names is None:
        provider_names = [p["name"] for p in list_providers()]

    if image is None:
        print("Taking benchmark screenshot...")
        image = take_screenshot()

    print(f"\n{'Provider':<15} {'Run':<5} {'Elements':<10} {'Time (s)':<10}")
    print("-" * 42)

    summary = {}
    for name in provider_names:
        times, counts = [], []
        for run in range(1, runs + 1):
            try:
                result = parse_screen(image=image, provider=name)
                times.append(result["parse_time"])
                counts.append(result["count"])
                print(
                    f"{name:<15} {run:<5} {result['count']:<10} {result['parse_time']:<10.3f}"
                )
            except Exception as e:
                print(f"{name:<15} {run:<5} ERROR: {e}")
                times.append(None)
                counts.append(0)

        valid_times = [t for t in times if t is not None]
        summary[name] = {
            "avg_time": (
                round(sum(valid_times) / len(valid_times), 3) if valid_times else None
            ),
            "avg_elements": round(sum(counts) / len(counts), 1),
        }

    print("\n── Summary ──")
    print(f"{'Provider':<15} {'Avg time (s)':<15} {'Avg elements'}")
    print("-" * 42)
    for name, stats in summary.items():
        t = f"{stats['avg_time']:.3f}" if stats["avg_time"] else "ERROR"
        print(f"{name:<15} {t:<15} {stats['avg_elements']}")

    return summary
