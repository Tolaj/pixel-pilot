# perception/providers/ax.py
from PIL import Image
from .base_provider import LazyProvider
from ..registry import register

INTERACTIVE_ROLES = {
    "AXButton",
    "AXCheckBox",
    "AXRadioButton",
    "AXTextField",
    "AXTextArea",
    "AXComboBox",
    "AXPopUpButton",
    "AXMenuItem",
    "AXMenuBarItem",
    "AXLink",
    "AXSlider",
    "AXTab",
    "AXCell",
    "AXStaticText",
    "AXImage",
    "AXToolbar",
}


@register
class AXProvider(LazyProvider):
    name = "ax"
    description = (
        "macOS Accessibility API — zero-latency, semantic roles, no vision model"
    )

    def _load(self):
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCopyAttributeValue,
                kAXChildrenAttribute,
                kAXRoleAttribute,
                kAXTitleAttribute,
                kAXValueAttribute,
                kAXPositionAttribute,
                kAXSizeAttribute,
            )

            self._ax_syms = dict(
                CreateSystemWide=AXUIElementCreateSystemWide,
                CopyAttr=AXUIElementCopyAttributeValue,
                kChildren=kAXChildrenAttribute,
                kRole=kAXRoleAttribute,
                kTitle=kAXTitleAttribute,
                kValue=kAXValueAttribute,
                kPosition=kAXPositionAttribute,
                kSize=kAXSizeAttribute,
            )
        except ImportError:
            raise ImportError(
                "pyobjc not installed.\n"
                "Run: pip install pyobjc-framework-ApplicationServices"
            )

    def _get_attr(self, el, attr):
        err, val = self._ax_syms["CopyAttr"](el, attr, None)
        return val if err == 0 else None

    def _walk(self, el, elements, depth=0, max_depth=12):
        if depth > max_depth:
            return

        s = self._ax_syms
        role = self._get_attr(el, s["kRole"]) or ""
        title = self._get_attr(el, s["kTitle"]) or ""
        value = self._get_attr(el, s["kValue"]) or ""
        pos = self._get_attr(el, s["kPosition"])
        size = self._get_attr(el, s["kSize"])

        label = (title or value or "").strip()

        if role in INTERACTIVE_ROLES and label and pos and size:
            x, y = pos.x, pos.y
            w, h = size.width, size.height
            if w > 0 and h > 0:
                elements.append(
                    {
                        "id": len(elements),
                        "label": label,
                        "role": role,
                        "center": [int(x + w / 2), int(y + h / 2)],
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                        "conf": 1.0,
                        "source": self.name,
                    }
                )

        children = self._get_attr(el, s["kChildren"]) or []
        for child in children:
            self._walk(child, elements, depth + 1)

    def parse(self, image: Image.Image) -> list[dict]:
        self._ensure_loaded()
        root = self._ax_syms["CreateSystemWide"]()
        elements = []
        self._walk(root, elements)
        return elements
