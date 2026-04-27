"""
tests/test_generator.py
Unit tests for the StableDiffusionGenerator pipeline wrapper.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipe():
    """Return a MagicMock simulating a diffusers pipeline."""
    pipe = MagicMock()
    pipe.scheduler = MagicMock()
    pipe.scheduler.config = {}
    pipe.unet = MagicMock()
    pipe.config = MagicMock()

    # Simulate pipeline call returning 2 dummy images
    dummy_images = [Image.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10)) for i in range(2)]
    pipe.return_value = MagicMock(images=dummy_images)
    return pipe


@pytest.fixture
def generator(mock_pipe):
    with patch("src.pipeline.generator.DiffusionPipeline") as mock_cls, \
         patch("src.pipeline.generator.StableDiffusionXLPipeline") as mock_xl:
        mock_xl.from_pretrained.return_value = mock_pipe
        mock_cls.from_pretrained.return_value = mock_pipe

        from src.pipeline.generator import StableDiffusionGenerator
        gen = StableDiffusionGenerator.__new__(StableDiffusionGenerator)
        gen.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        gen.device = "cpu"
        gen.pipe = mock_pipe
        return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_list_of_pil_images(self, generator, mock_pipe):
        images = generator.generate(
            prompt="A beautiful mountain",
            num_images=2,
            height=64,
            width=64,
            num_inference_steps=1,
        )
        assert isinstance(images, list)
        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)

    def test_with_seed_calls_pipe_with_generators(self, generator, mock_pipe):
        generator.generate(
            prompt="test prompt",
            seed=42,
            num_images=1,
            height=64,
            width=64,
            num_inference_steps=1,
        )
        mock_pipe.assert_called_once()
        call_kwargs = mock_pipe.call_args.kwargs
        assert "generator" in call_kwargs

    def test_without_seed_uses_random(self, generator, mock_pipe):
        generator.generate(
            prompt="test prompt",
            seed=None,
            num_images=1,
            height=64,
            width=64,
            num_inference_steps=1,
        )
        mock_pipe.assert_called_once()

    def test_negative_prompt_passed(self, generator, mock_pipe):
        generator.generate(
            prompt="a dog",
            negative_prompt="blurry",
            num_images=1,
            num_inference_steps=1,
        )
        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs.get("negative_prompt") == "blurry"


class TestSaveGrid:
    def test_saves_file(self, generator, tmp_path):
        images = [Image.new("RGB", (64, 64)) for _ in range(4)]
        out_path = tmp_path / "grid.png"
        generator.save_grid(images, out_path)
        assert out_path.exists()

    def test_creates_parent_dirs(self, generator, tmp_path):
        images = [Image.new("RGB", (64, 64)) for _ in range(2)]
        out_path = tmp_path / "nested" / "dir" / "grid.png"
        generator.save_grid(images, out_path)
        assert out_path.exists()


class TestSaveAll:
    def test_saves_all_images(self, generator, tmp_path):
        images = [Image.new("RGB", (32, 32)) for _ in range(3)]
        paths = generator.save_all(images, tmp_path, prefix="test")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_filenames_have_prefix(self, generator, tmp_path):
        images = [Image.new("RGB", (32, 32))]
        paths = generator.save_all(images, tmp_path, prefix="myprefix")
        assert "myprefix" in paths[0].name
