"""Tests for dataset loading and transforms."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from src.data import (
    VisionLanguageDataset,
    build_transforms,
    generate_dummy_data,
    load_annotations,
    split_annotations,
)


class DummyTokenizer:
    """Minimal tokenizer stand-in."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def __call__(self, text: str, **kwargs: dict) -> dict:
        ids = list(range(min(len(text.split()), kwargs.get("max_length", 16))))
        pad_len = kwargs.get("max_length", 16) - len(ids)
        ids += [self.pad_token_id] * pad_len
        mask = [1 if i < len(text.split()) else 0 for i in range(kwargs.get("max_length", 16))]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def test_build_transforms() -> None:
    """Transform pipeline should produce a torch tensor."""
    transform = build_transforms(["resize", "normalize"], image_size=32)
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    tensor = transform(img)
    assert tensor.shape == (3, 32, 32)


def test_load_annotations(tmp_path) -> None:
    """Annotations should load from JSON."""
    path = tmp_path / "annotations.json"
    data = [{"image_id": "1", "caption": "hello"}]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    loaded = load_annotations(path)
    assert loaded == data


def test_split_annotations() -> None:
    """Train/val split should respect val_size."""
    ann = [{"id": i} for i in range(100)]
    train, val = split_annotations(ann, val_size=0.2, seed=42)
    assert len(train) == 80
    assert len(val) == 20


def test_vision_language_dataset(tmp_path) -> None:
    """Dataset should return tensors and metadata."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    img.save(image_dir / "00001.jpg")

    annotations = [{"image_id": "00001", "image_path": "images/00001.jpg", "caption": "a photo"}]
    tokenizer = DummyTokenizer()
    transform = build_transforms(["resize", "normalize"], 32)
    dataset = VisionLanguageDataset(
        annotations=annotations,
        image_dir=tmp_path,
        tokenizer=tokenizer,
        transform=transform,
        max_length=16,
    )
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 32, 32)
    assert sample["input_ids"].shape == (16,)


def test_generate_dummy_data(tmp_path) -> None:
    """Dummy data generator should create images and annotations."""
    path = generate_dummy_data(tmp_path, num_samples=5, image_size=32)
    assert path.exists()
    assert len(load_annotations(path)) == 5
    assert (tmp_path / "images").exists()
