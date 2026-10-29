"""Multimodal data curation for efficient VLM training.

Inspired by recent findings that concise, well-aligned training data improves
both inference efficiency (lower Cost-of-Pass) and final model quality with
fewer resources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def caption_concision_score(text: str, tokenizer: Any) -> float:
    """Score caption brevity: lower tokens-per-word means more concise.

    Args:
        text: Caption text.
        tokenizer: Text tokenizer.

    Returns:
        Tokens per word (lower is more concise).
    """
    words = [w for w in text.split() if w.strip()]
    if not words:
        return float("inf")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) / len(words)


class CLIPSimilarityScorer:
    """Compute image-text similarity using a CLIP-style model.

    Falls back gracefully if transformers / CLIP is unavailable.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        """Initialize CLIP model and processor.

        Args:
            model_name: Hugging Face CLIP model name.
        """
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required for CLIP similarity") from exc

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.inference_mode()
    def score(self, image: Image.Image, text: str) -> float:
        """Return cosine similarity between image and text embeddings.

        Args:
            image: PIL image.
            text: Caption text.

        Returns:
            Similarity score (higher means more aligned).
        """
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return float(logits_per_image.item())


def simple_similarity_fallback(image: Image.Image, text: str) -> float:
    """Lightweight fallback similarity when CLIP is unavailable.

    Uses image brightness variance as a weak proxy for visual informativeness
    and counts caption length. Not discriminative, but avoids hard dependency.

    Args:
        image: PIL image.
        text: Caption text.

    Returns:
        Dummy similarity score.
    """
    arr = np.array(image.convert("L"))
    contrast = float(np.std(arr))
    word_count = len([w for w in text.split() if w.strip()])
    return contrast + min(word_count, 20)


def curate_annotations(
    annotations: list[dict[str, Any]],
    image_dir: Path,
    tokenizer: Any,
    keep_top_fraction: float = 0.8,
    min_similarity: float | None = None,
    max_tokens_per_word: float | None = None,
    clip_model: str | None = "openai/clip-vit-base-patch32",
) -> list[dict[str, Any]]:
    """Filter VLM annotations by caption concision and image-text alignment.

    Args:
        annotations: List of annotation dicts.
        image_dir: Base directory for images.
        tokenizer: Text tokenizer.
        keep_top_fraction: Fraction to keep by combined score.
        min_similarity: Optional CLIP similarity threshold.
        max_tokens_per_word: Optional verbosity cutoff.
        clip_model: CLIP model name, or None to use fallback.

    Returns:
        Curated annotation list.
    """
    scorer: CLIPSimilarityScorer | None = None
    if clip_model is not None:
        try:
            scorer = CLIPSimilarityScorer(clip_model)
        except Exception:  # pragma: no cover - CLIP may fail to load
            scorer = None

    records = []
    for ann in annotations:
        caption = ann.get("caption", ann.get("answer", ""))
        image_path = image_dir / ann.get("image_path", f"{ann['image_id']}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        concision = caption_concision_score(caption, tokenizer)
        if max_tokens_per_word is not None and concision > max_tokens_per_word:
            continue

        if scorer is not None:
            similarity = scorer.score(image, caption)
        else:
            similarity = simple_similarity_fallback(image, caption)

        if min_similarity is not None and similarity < min_similarity:
            continue

        records.append(
            {
                **ann,
                "concision_score": concision,
                "similarity_score": similarity,
            }
        )

    if not records:
        return records

    # Normalize scores and compute a combined rank (lower concision is better,
    # higher similarity is better).
    concisions = np.array([r["concision_score"] for r in records])
    similarities = np.array([r["similarity_score"] for r in records])

    norm_concision = (concisions - concisions.min()) / (concisions.max() - concisions.min() + 1e-8)
    norm_similarity = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
    combined = norm_concision - norm_similarity  # lower is better

    threshold = np.percentile(combined, 100 * (1 - keep_top_fraction))
    curated = [
        {k: v for k, v in r.items() if k not in {"concision_score", "similarity_score"}}
        for r, score in zip(records, combined, strict=False)
        if score <= threshold
    ]

    return curated
