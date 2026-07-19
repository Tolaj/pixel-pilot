"""
Map visual token indices to screen coordinates.

Accounts for patch_size, spatial_merge_size, resize scaling, and Retina display.
Supports both Qwen2-VL (patch_size=14) and Qwen3-VL (patch_size=16).
"""

from typing import Tuple, Callable


def build_patch_map(
    image_grid_thw,
    resized_size,
    original_size,
    retina_scale=2.0,
    patch_size=14,
    spatial_merge_size=2,
) -> Tuple[Callable[[int], Tuple[float, float]], Tuple[int, int]]:
    """
    Build a mapping from visual token index to screen coordinates.

    Args:
        image_grid_thw: (temporal, height_patches, width_patches) from processor.
        resized_size: (height, width) of the image fed to the model.
        original_size: (height, width) of the original screenshot.
        retina_scale: Display scaling factor (2.0 for Retina).
        patch_size: ViT patch size (14 for Qwen2-VL, 16 for Qwen3-VL).
        spatial_merge_size: How many patches are merged per dimension (typically 2).

    Returns:
        (patch_map_fn, grid_shape)
        patch_map_fn(token_idx) -> (screen_x, screen_y)
        grid_shape = (merged_rows, merged_cols)
    """
    _, h_patches, w_patches = image_grid_thw
    merged_rows = h_patches // spatial_merge_size
    merged_cols = w_patches // spatial_merge_size

    merged_patch_px = patch_size * spatial_merge_size

    res_h, res_w = resized_size
    orig_h, orig_w = original_size
    scale_x = orig_w / res_w
    scale_y = orig_h / res_h

    def patch_map_fn(token_idx: int) -> Tuple[float, float]:
        row = token_idx // merged_cols
        col = token_idx % merged_cols
        cx_resized = (col + 0.5) * merged_patch_px
        cy_resized = (row + 0.5) * merged_patch_px
        screen_x = (cx_resized * scale_x) / retina_scale
        screen_y = (cy_resized * scale_y) / retina_scale
        return (screen_x, screen_y)

    return patch_map_fn, (merged_rows, merged_cols)
