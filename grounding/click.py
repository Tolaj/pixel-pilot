"""Execute a click at screen coordinates via pyautogui."""

from typing import Tuple
from pathlib import Path


def click(
    screen_x: float,
    screen_y: float,
    dry_run: bool = False,
    debug_path: str = "debug/click_preview.png",
) -> Tuple[float, float]:
    """
    Click at screen coordinates, or save a preview in dry-run mode.

    Returns:
        (screen_x, screen_y) that was clicked/previewed.
    """
    if dry_run:
        import subprocess
        import tempfile
        from PIL import Image, ImageDraw

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        subprocess.run(["screencapture", "-x", tmp.name], check=True)
        img = Image.open(tmp.name)

        draw = ImageDraw.Draw(img)
        px, py = int(screen_x * 2), int(screen_y * 2)
        r = 20
        draw.ellipse([px - r, py - r, px + r, py + r], outline="lime", width=4)
        draw.line([px - r, py, px + r, py], fill="lime", width=3)
        draw.line([px, py - r, px, py + r], fill="lime", width=3)

        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(debug_path)
        print(f"[dry-run] Would click at ({screen_x:.0f}, {screen_y:.0f}). Preview: {debug_path}")
    else:
        import pyautogui
        pyautogui.click(screen_x, screen_y)
        print(f"Clicked at ({screen_x:.0f}, {screen_y:.0f})")

    return (screen_x, screen_y)
