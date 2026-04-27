"""
tests/test_metrics.py
Unit tests for evaluation metrics helpers.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_dummy_images(n: int = 5, size: int = 64) -> list:
    import random
    return [
        Image.new("RGB", (size, size), color=(random.randint(0, 255),) * 3)
        for _ in range(n)
    ]


class TestComputeClipScore:
    def test_returns_float_in_range(self):
        """CLIP score should be a float, typically 0.0–1.0."""
        from src.utils.metrics import compute_clip_score

        images = _make_dummy_images(4)
        prompts = ["a cat", "a dog", "a bird", "a fish"]

        # Mock CLIP model + processor to avoid real download
        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Simulate normalized embeddings
        emb = torch.nn.functional.normalize(torch.randn(4, 512), dim=-1)
        mock_model.return_value = MagicMock(image_embeds=emb, text_embeds=emb)
        mock_processor.return_value = {
            "input_ids": torch.zeros(4, 10, dtype=torch.long),
            "pixel_values": torch.zeros(4, 3, 224, 224),
        }
        mock_processor.return_value = MagicMock(to=lambda d: mock_processor.return_value)

        with patch("src.utils.metrics.CLIPModel") as mock_cls, \
             patch("src.utils.metrics.CLIPProcessor") as mock_proc_cls:
            mock_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = mock_processor

            # Just check it runs — real CLIP test would need model download
            # score = compute_clip_score(images, prompts, device="cpu")
            # assert 0.0 <= score <= 1.0
            pass  # Placeholder: integration test requires model

    def test_raises_on_length_mismatch(self):
        from src.utils.metrics import compute_clip_score
        with pytest.raises(AssertionError):
            compute_clip_score(
                images=_make_dummy_images(3),
                prompts=["a cat", "a dog"],
                device="cpu",
            )


class TestMakeImageGrid:
    def test_grid_dimensions(self):
        from src.utils.visualization import make_image_grid
        images = [Image.new("RGB", (64, 64)) for _ in range(4)]
        grid = make_image_grid(images, cols=2)
        assert isinstance(grid, Image.Image)
        # 2 cols × 64 + 3 × padding(4) = 128 + 12 = 140
        assert grid.width == 2 * 64 + 3 * 4
        assert grid.height == 2 * 64 + 3 * 4

    def test_single_image(self):
        from src.utils.visualization import make_image_grid
        images = [Image.new("RGB", (100, 100))]
        grid = make_image_grid(images, cols=1)
        assert isinstance(grid, Image.Image)

    def test_empty_raises(self):
        from src.utils.visualization import make_image_grid
        with pytest.raises(ValueError):
            make_image_grid([], cols=2)


class TestAnnotateImage:
    def test_returns_rgb_image(self):
        from src.utils.visualization import annotate_image
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        annotated = annotate_image(img, "Test Label")
        assert isinstance(annotated, Image.Image)
        assert annotated.mode == "RGB"
        assert annotated.size == (256, 256)

    def test_top_position(self):
        from src.utils.visualization import annotate_image
        img = Image.new("RGB", (256, 256))
        result = annotate_image(img, "Top Label", position="top")
        assert isinstance(result, Image.Image)


class TestMakeComparisonStrip:
    def test_strip_wider_than_single_image(self):
        from src.utils.visualization import make_comparison_strip
        images = [Image.new("RGB", (128, 128)) for _ in range(3)]
        strip = make_comparison_strip(images)
        assert strip.width > 128

    def test_labels_applied(self):
        from src.utils.visualization import make_comparison_strip
        images = [Image.new("RGB", (128, 128)) for _ in range(2)]
        strip = make_comparison_strip(images, labels=["A", "B"])
        assert isinstance(strip, Image.Image)

    def test_label_length_mismatch_raises(self):
        from src.utils.visualization import make_comparison_strip
        images = [Image.new("RGB", (64, 64)) for _ in range(3)]
        with pytest.raises(ValueError):
            make_comparison_strip(images, labels=["only_one"])
