"""Evaluation metrics and inference for the vision-language model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import VisionLanguageDataset, build_transforms, load_annotations
from src.logging_utils import get_logger, setup_logging
from src.models import load_components
from src.utils import get_device, load_config

logger = get_logger(__name__)


def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute corpus BLEU score.

    Args:
        predictions: List of predicted strings.
        references: List of reference lists.

    Returns:
        Dictionary with BLEU scores.
    """
    try:
        from sacrebleu import corpus_bleu
    except ImportError as exc:
        raise RuntimeError("sacrebleu is required for BLEU") from exc

    score = corpus_bleu(predictions, list(zip(*references, strict=False)))
    return {"bleu": score.score}


def compute_rouge(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute ROUGE-L score.

    Args:
        predictions: List of predicted strings.
        references: List of reference lists.

    Returns:
        Dictionary with ROUGE-L score.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("rouge-score is required for ROUGE") from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, refs in zip(predictions, references, strict=False):
        best = max(scorer.score(pred, ref)["rougeL"].fmeasure for ref in refs)
        scores.append(best)
    return {"rougeL": sum(scores) / len(scores)}


def evaluate_model(
    model_path: str,
    config: dict[str, Any],
    split: str = "val",
) -> dict[str, Any]:
    """Run evaluation and return metrics plus per-sample results.

    Args:
        model_path: Path to checkpoint.
        config: Evaluation config.
        split: Dataset split to evaluate.

    Returns:
        Dictionary with metrics and samples.
    """
    device = get_device()
    model, tokenizer = load_components(config["model"], device=device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    annotations = load_annotations(config["data"]["annotations"])
    # Simple deterministic split.
    split_idx = int(len(annotations) * (1 - config["data"].get("val_size", 0.1)))
    eval_ann = annotations[split_idx:] if split == "val" else annotations[:split_idx]

    image_dir = Path(config["data"]["image_dir"])
    transform = build_transforms(
        config["data"].get("transforms", ["resize", "normalize"]),
        config["data"]["image_size"],
    )
    dataset = VisionLanguageDataset(
        annotations=eval_ann,
        image_dir=image_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_length=config["model"]["max_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
    )

    predictions: list[str] = []
    references: list[list[str]] = []
    samples: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        generated = model.generate(
            pixel_values=pixel_values,
            tokenizer=tokenizer,
            max_length=config["model"].get("max_length", 128),
        )
        for i, pred in enumerate(generated):
            ref = batch["metadata"]["text"][i] if isinstance(batch["metadata"]["text"], list) else batch["metadata"]["text"]
            predictions.append(pred)
            references.append([ref])
            samples.append(
                {
                    "image_id": batch["metadata"]["image_id"][i],
                    "prediction": pred,
                    "reference": ref,
                }
            )

    metrics: dict[str, float] = {}
    requested = config.get("metrics", ["bleu", "rouge"])
    if "bleu" in requested:
        metrics.update(compute_bleu(predictions, references))
    if "rouge" in requested:
        metrics.update(compute_rouge(predictions, references))

    return {"metrics": metrics, "samples": samples}


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate multimodal VLM")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to eval config")
    parser.add_argument("--output_dir", help="Directory to write results")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    results = evaluate_model(args.model_path, config)

    logger.info("Metrics: %s", results["metrics"])

    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_dir / "eval_results.json")


if __name__ == "__main__":
    main()
