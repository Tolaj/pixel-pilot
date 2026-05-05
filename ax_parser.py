# ax_parser.py  — uses macOS Accessibility API via pyobjc
import time
from AppKit import NSWorkspace
from ApplicationServices import (
    AXUIElementCreateSystemWide,
    AXUIElementCopyAttributeValue,
    kAXChildrenAttribute,
    kAXRoleAttribute,
    kAXTitleAttribute,
    kAXValueAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXEnabledAttribute,
)
from CoreFoundation import CFTypeRef
import Quartz

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
}


def _get_attr(el, attr):
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == 0 else None


def _walk(el, elements, depth=0, max_depth=10):
    if depth > max_depth:
        return

    role = _get_attr(el, kAXRoleAttribute) or ""
    title = _get_attr(el, kAXTitleAttribute) or ""
    value = _get_attr(el, kAXValueAttribute) or ""
    pos = _get_attr(el, kAXPositionAttribute)
    size = _get_attr(el, kAXSizeAttribute)

    label = title or value or ""

    if role in INTERACTIVE_ROLES and label and pos and size:
        x, y = pos.x, pos.y
        w, h = size.width, size.height
        cx, cy = int(x + w / 2), int(y + h / 2)
        elements.append(
            {
                "id": len(elements),
                "label": label,
                "role": role,
                "center": [cx, cy],
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "conf": 1.0,  # AX is ground truth, not probabilistic
            }
        )

    children = _get_attr(el, kAXChildrenAttribute) or []
    for child in children:
        _walk(child, elements, depth + 1)


def ax_parse() -> list[dict]:
    root = AXUIElementCreateSystemWide()
    elements = []
    _walk(root, elements)
    return elements
