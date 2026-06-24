"""Outlier and false-positive analysis for VLM outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import VisionLanguageDataset, build_transforms, load_annotations
from src.logging_utils import get_logger, setup_logging
from src.models import VisionLanguageModel, load_components
from src.utils import get_device, load_config

logger = get_logger(__name__)


def compute_per_sample_loss(
    model: VisionLanguageModel,
    dataloader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Compute per-sample cross-entropy loss.

    Args:
        model: VLM instance.
        dataloader: Evaluation dataloader.
        device: Torch device.

    Returns:
        List of records with metadata and loss.
    """
    model.eval()
    results = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for batch in tqdm(dataloader, desc="Computing per-sample losses"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.inference_mode():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs["logits"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        masked_losses = (losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

        for i in range(pixel_values.size(0)):
            results.append(
                {
                    "image_id": batch["metadata"]["image_id"][i],
                    "reference": batch["metadata"]["text"][i],
                    "loss": masked_losses[i].item(),
                }
            )

    return results


def detect_outliers(records: list[dict[str, Any]], percentile: float = 95.0) -> list[dict[str, Any]]:
    """Flag records whose loss exceeds the given percentile.

    Args:
        records: Per-sample loss records.
        percentile: Loss percentile threshold.

    Returns:
        List of outlier records with threshold.
    """
    losses = [r["loss"] for r in records]
    threshold = float(np.percentile(losses, percentile))
    outliers = [{**r, "threshold": threshold} for r in records if r["loss"] > threshold]
    outliers.sort(key=lambda x: x["loss"], reverse=True)
    return outliers


def detect_repeated_tokens(text: str, min_repeats: int = 4) -> bool:
    """Detect repetitive generation (e.g. model stuck on a token).

    Args:
        text: Generated text.
        min_repeats: Minimum consecutive repeats to flag.

    Returns:
        True if repetitive patterns detected.
    """
    tokens = text.split()
    if len(tokens) < min_repeats:
        return False
    for i in range(len(tokens) - min_repeats + 1):
        window = tokens[i : i + min_repeats]
        if len(set(window)) == 1:
            return True
    return False


def lexical_overlap(prediction: str, reference: str) -> float:
    """Compute token-level overlap between prediction and reference.

    Args:
        prediction: Predicted text.
        reference: Reference text.

    Returns:
        Intersection-over-union score.
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    intersection = len(pred_tokens & ref_tokens)
    union = len(pred_tokens | ref_tokens)
    return intersection / union if union > 0 else 0.0


def analyze_false_positives(
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.3,
    min_confidence: float = 0.8,
) -> dict[str, Any]:
    """Separate model issues from data issues.

    Args:
        predictions: List of prediction records.
        iou_threshold: IoU below which a prediction is considered wrong.
        min_confidence: Confidence threshold for high-confidence wrong predictions.

    Returns:
        Analysis report dictionary.
    """
    model_problems = []
    data_problems = []

    for pred in predictions:
        iou = lexical_overlap(pred["prediction"], pred["reference"])
        is_wrong = iou < iou_threshold
        is_repetitive = detect_repeated_tokens(pred["prediction"])
        # Use average softmax confidence as proxy.
        confidence = pred.get("confidence", 0.0)

        if is_wrong:
            if confidence >= min_confidence or is_repetitive:
                model_problems.append(
                    {
                        **pred,
                        "issue": "high_confidence_wrong" if not is_repetitive else "repetitive_generation",
                        "iou": iou,
                    }
                )
            else:
                data_problems.append({**pred, "issue": "annotation_mismatch_or_noisy_label", "iou": iou})

    report = {
        "total": len(predictions),
        "model_problems_count": len(model_problems),
        "data_problems_count": len(data_problems),
        "model_problems": model_problems[:50],
        "data_problems": data_problems[:50],
        "avg_iou": mean([lexical_overlap(p["prediction"], p["reference"]) for p in predictions]),
        "common_prediction_tokens": Counter(
            " ".join(p["prediction"] for p in predictions).lower().split()
        ).most_common(20),
    }
    return report


def run_analysis(
    model_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run full outlier + false-positive analysis.

    Args:
        model_path: Path to checkpoint.
        config: Full project config.

    Returns:
        Analysis report dictionary.
    """
    device = get_device()
    model, tokenizer = load_components(config["model"], device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    annotations = load_annotations(config["data"]["annotations"])
    split_idx = int(len(annotations) * (1 - config["data"].get("val_size", 0.1)))
    eval_ann = annotations[split_idx:]

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

    logger.info("Computing per-sample losses")
    loss_records = compute_per_sample_loss(model, dataloader, device)
    outliers = detect_outliers(
        loss_records,
        percentile=config.get("analysis", {}).get("outlier_loss_percentile", 95),
    )

    logger.info("Generating predictions for false-positive analysis")
    predictions = []
    for batch in tqdm(dataloader, desc="Generating"):
        pixel_values = batch["pixel_values"].to(device)
        generated = model.generate(
            pixel_values=pixel_values,
            tokenizer=tokenizer,
            max_length=config["model"].get("max_length", 128),
        )
        for i, pred in enumerate(generated):
            predictions.append(
                {
                    "image_id": batch["metadata"]["image_id"][i],
                    "prediction": pred,
                    "reference": batch["metadata"]["text"][i],
                    "confidence": 0.0,  # Could be populated from softmax scores.
                }
            )

    fp_report = analyze_false_positives(
        predictions,
        iou_threshold=config.get("analysis", {}).get("fp_iou_threshold", 0.3),
        min_confidence=config.get("analysis", {}).get("min_confidence", 0.8),
    )

    return {
        "outliers": outliers,
        "false_positive_report": fp_report,
    }


def main() -> None:
    """CLI entry point for analysis."""
    parser = argparse.ArgumentParser(description="Analyse VLM outputs")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", default="outputs/analysis", help="Output directory")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    report = run_analysis(args.model_path, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "error_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Analysis report saved to %s", output_dir / "error_report.json")


if __name__ == "__main__":
    main()
