"""Unit tests for patch_map: verify token->screen coordinate mapping."""

import pytest
from grounding.patch_map import build_patch_map


class TestBuildPatchMapQwen2:
    """Test patch map with Qwen2-VL params (patch_size=14, merge=2 -> 28px tokens)."""

    def test_single_patch(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 2, 2),
            resized_size=(56, 56),
            original_size=(56, 56),
            retina_scale=1.0,
            patch_size=14,
            spatial_merge_size=2,
        )
        assert shape == (1, 1)
        assert fn(0) == pytest.approx((14.0, 14.0))

    def test_2x2_grid_corners(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 4, 4),
            resized_size=(112, 112),
            original_size=(112, 112),
            retina_scale=1.0,
            patch_size=14,
            spatial_merge_size=2,
        )
        assert shape == (2, 2)
        assert fn(0) == pytest.approx((14.0, 14.0))
        assert fn(1) == pytest.approx((42.0, 14.0))
        assert fn(2) == pytest.approx((14.0, 42.0))
        assert fn(3) == pytest.approx((42.0, 42.0))

    def test_retina_scaling(self):
        fn, _ = build_patch_map(
            image_grid_thw=(1, 4, 4),
            resized_size=(112, 112),
            original_size=(112, 112),
            retina_scale=2.0,
            patch_size=14,
            spatial_merge_size=2,
        )
        assert fn(0) == pytest.approx((7.0, 7.0))

    def test_resize_scaling(self):
        fn, _ = build_patch_map(
            image_grid_thw=(1, 4, 4),
            resized_size=(112, 112),
            original_size=(224, 224),
            retina_scale=1.0,
            patch_size=14,
            spatial_merge_size=2,
        )
        assert fn(0) == pytest.approx((28.0, 28.0))


class TestBuildPatchMapQwen3:
    """Test patch map with Qwen3-VL params (patch_size=16, merge=2 -> 32px tokens)."""

    def test_single_patch(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 2, 2),
            resized_size=(64, 64),
            original_size=(64, 64),
            retina_scale=1.0,
            patch_size=16,
            spatial_merge_size=2,
        )
        assert shape == (1, 1)
        assert fn(0) == pytest.approx((16.0, 16.0))

    def test_2x2_grid_corners(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 4, 4),
            resized_size=(128, 128),
            original_size=(128, 128),
            retina_scale=1.0,
            patch_size=16,
            spatial_merge_size=2,
        )
        assert shape == (2, 2)
        assert fn(0) == pytest.approx((16.0, 16.0))
        assert fn(1) == pytest.approx((48.0, 16.0))
        assert fn(2) == pytest.approx((16.0, 48.0))
        assert fn(3) == pytest.approx((48.0, 48.0))

    def test_realistic_512x320(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 20, 32),
            resized_size=(320, 512),
            original_size=(320, 512),
            retina_scale=1.0,
            patch_size=16,
            spatial_merge_size=2,
        )
        assert shape == (10, 16)
        assert fn(88) == pytest.approx((272.0, 176.0))

    def test_retina_and_resize(self):
        fn, _ = build_patch_map(
            image_grid_thw=(1, 4, 4),
            resized_size=(128, 128),
            original_size=(256, 256),
            retina_scale=2.0,
            patch_size=16,
            spatial_merge_size=2,
        )
        assert fn(0) == pytest.approx((16.0, 16.0))

    def test_total_tokens(self):
        fn, shape = build_patch_map(
            image_grid_thw=(1, 20, 32),
            resized_size=(320, 512),
            original_size=(640, 1024),
            retina_scale=2.0,
            patch_size=16,
            spatial_merge_size=2,
        )
        rows, cols = shape
        assert rows == 10
        assert cols == 16
        for i in range(rows * cols):
            x, y = fn(i)
            assert x >= 0 and y >= 0
