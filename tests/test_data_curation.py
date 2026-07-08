"""Tests for multimodal data curation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.data_curation import caption_concision_score, curate_annotations


class DummyTokenizer:
    """Minimal tokenizer for curation tests."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


def test_caption_concision_score() -> None:
    """Concision score should penalise verbose captions."""
    tokenizer = DummyTokenizer()
    short = caption_concision_score("A cat.", tokenizer)
    long = caption_concision_score("A fluffy domesticated feline animal sitting down.", tokenizer)
    assert short < long


def test_curate_annotations(tmp_path) -> None:
    """Curation should retain concise examples and drop verbose ones."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    annotations = []
    for i in range(4):
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        filename = f"{i:05d}.jpg"
        img.save(image_dir / filename)
        annotations.append(
            {
                "image_id": f"{i:05d}",
                "image_path": str(image_dir / filename),
                "caption": "a" if i < 2 else "a b c d e f g h i j k l m n o p",
            }
        )

    tokenizer = DummyTokenizer()
    curated = curate_annotations(
        annotations=annotations,
        image_dir=image_dir,
        tokenizer=tokenizer,
        keep_top_fraction=0.5,
        max_tokens_per_word=5.0,
        clip_model=None,  # use fallback
    )
    assert len(curated) <= 2
    assert all("concision_score" not in ann for ann in curated)
