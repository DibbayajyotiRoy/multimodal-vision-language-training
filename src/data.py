"""Dataset and transforms for vision-language training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import resolve_path


def build_transforms(transform_config: list[str], image_size: int) -> transforms.Compose:
    """Build a torchvision transform pipeline from a list of transform names.

    Args:
        transform_config: List of transform names (e.g. resize, normalize, random_horizontal_flip).
        image_size: Target image size.

    Returns:
        Composed torchvision transforms.
    """
    ops: list[Any] = []
    for name in transform_config:
        name = name.lower()
        if name == "resize":
            ops.append(transforms.Resize((image_size, image_size)))
        elif name == "center_crop":
            ops.append(transforms.CenterCrop(image_size))
        elif name == "random_horizontal_flip":
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        elif name == "random_rotation":
            ops.append(transforms.RandomRotation(degrees=10))
        elif name == "color_jitter":
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        elif name == "to_tensor":
            ops.append(transforms.ToTensor())
        elif name == "normalize":
            ops.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
    if "to_tensor" not in [t.lower() for t in transform_config]:
        ops.insert(0, transforms.ToTensor())
    if "normalize" not in [t.lower() for t in transform_config]:
        ops.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)


class VisionLanguageDataset(Dataset):
    """PyTorch dataset for image-caption / visual-question-answer pairs."""

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        image_dir: Path,
        tokenizer: Any,
        transform: transforms.Compose,
        max_length: int = 128,
    ) -> None:
        """Initialize dataset.

        Args:
            annotations: List of annotation dicts with image_path and caption/answer.
            image_dir: Base directory for images.
            tokenizer: Text tokenizer.
            transform: Image transform pipeline.
            max_length: Maximum tokenized text length.
        """
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single training example."""
        item = self.annotations[idx]
        image_path = self.image_dir / item.get("image_path", f"{item['image_id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)

        text = item.get("caption", item.get("answer", ""))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
            "metadata": {"image_id": item.get("image_id", idx), "text": text},
        }


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load annotations from JSON or JSONL.

    Args:
        path: Path to annotations file.

    Returns:
        List of annotation dictionaries.
    """
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def split_annotations(
    annotations: list[dict[str, Any]],
    val_size: float,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split annotations into train and validation sets.

    Args:
        annotations: Full annotation list.
        val_size: Fraction for validation.
        seed: Random seed.

    Returns:
        Tuple of (train_annotations, val_annotations).
    """
    rng = random.Random(seed)
    shuffled = annotations.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_size))
    return shuffled[:split_idx], shuffled[split_idx:]


def generate_dummy_data(output_dir: str | Path, num_samples: int = 100, image_size: int = 224) -> Path:
    """Generate a tiny dummy dataset for smoke tests.

    Args:
        output_dir: Where to write images and annotations.
        num_samples: Number of samples.
        image_size: Image dimensions.

    Returns:
        Path to annotations file.
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    for i in range(num_samples):
        arr = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        filename = f"{i:05d}.jpg"
        img.save(image_dir / filename)
        annotations.append(
            {
                "image_id": f"{i:05d}",
                "image_path": str(image_dir / filename),
                "caption": f"This is a dummy caption number {i}.",
            }
        )

    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
    return annotations_path


def main() -> None:
    """CLI to generate dummy data."""
    parser = argparse.ArgumentParser(description="Generate dummy VLM data")
    parser.add_argument("--output_dir", default="data/dummy")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    path = generate_dummy_data(args.output_dir, args.num_samples, args.image_size)
    print(f"Generated dummy dataset: {path}")


if __name__ == "__main__":
    main()
