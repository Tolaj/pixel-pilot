"""
Perception package -- modular screen understanding backends.

Usage:
    from pixel_pilot.perception import parse_screen, set_provider

    set_provider("ax")
    result = parse_screen()
"""

import os
import time
import pyautogui
from PIL import Image

from .registry import get, list_providers
from . import providers  # noqa: F401

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


def take_screenshot() -> Image.Image:
    return pyautogui.screenshot()


def parse_screen(image: Image.Image = None, provider: str = None) -> dict:
    """Capture and parse the current screen."""
    if image is None:
        image = take_screenshot()

    p = get(provider) if provider else _get_provider()

    start = time.time()
    elements = p.parse(image)
    elapsed = round(time.time() - start, 3)

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
