"""Staged training loop for the vision-language model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data import VisionLanguageDataset, build_transforms, load_annotations, split_annotations
from src.data_curation import curate_annotations
from src.logging_utils import get_logger, setup_logging
from src.models import (
    VisionLanguageModel,
    load_components,
    set_component_freezing,
    set_component_training,
)
from src.utils import (
    count_trainable_parameters,
    get_device,
    load_config,
    save_config,
    set_seed,
    setup_output_dir,
)

logger = get_logger(__name__)


def build_dataloader(
    annotations: list[dict[str, Any]],
    image_dir: Path,
    tokenizer: Any,
    transform: Any,
    config: dict[str, Any],
    stage_key: str,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for a training stage."""
    dataset = VisionLanguageDataset(
        annotations=annotations,
        image_dir=image_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_length=config["model"]["max_length"],
    )
    batch_size = config[stage_key]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    max_grad_norm: float,
    log_every: int,
    gradient_accumulation_steps: int,
) -> dict[str, float]:
    """Run one training epoch and return metrics."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    global_step = epoch * len(dataloader)

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=scaler is not None, dtype=torch.float16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps * pixel_values.size(0)
        total_samples += pixel_values.size(0)

        if writer is not None and global_step % log_every == 0:
            writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        progress.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        global_step += 1

    return {"loss": total_loss / total_samples}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs["loss"].item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)

    return {"loss": total_loss / total_samples}


def train_stage(
    config: dict[str, Any],
    stage_key: str,
    model: VisionLanguageModel | None = None,
    tokenizer: Any | None = None,
    resume_from: str | None = None,
) -> Path:
    """Run a single training stage.

    Args:
        config: Full project config.
        stage_key: 'stage1' or 'stage2'.
        model: Optional pre-loaded VLM (for stage 2 resume).
        tokenizer: Optional pre-loaded tokenizer.
        resume_from: Optional checkpoint path to resume.

    Returns:
        Path to the best saved checkpoint.
    """
    stage_cfg = config[stage_key]
    seed = stage_cfg.get("seed", 42)
    set_seed(seed)

    output_dir = setup_output_dir(config, stage_key)
    setup_logging(log_file=output_dir / "train.log")
    logger.info("Starting %s with config:\n%s", stage_key, config)
    save_config(config, output_dir / "config.yaml")

    device = get_device()
    if model is None or tokenizer is None:
        model, tokenizer = load_components(config["model"], device=device)

    # Apply staged freezing.
    freeze_components = stage_cfg.get("freeze_components", [])
    trainable_components = stage_cfg.get("trainable_components", [])
    set_component_freezing(model, freeze_components)
    set_component_training(model, trainable_components)

    trainable, total = count_trainable_parameters(model)
    logger.info(
        "Stage %s trainable params: %d / %d (%.2f%%)",
        stage_key, trainable, total, 100 * trainable / total,
    )

    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    annotations = load_annotations(config["data"]["annotations"])
    image_dir = Path(config["data"]["image_dir"])

    # Optional data-centric curation: keep concise, image-aligned examples.
    curation_cfg = config.get("data", {}).get("curation", {})
    if curation_cfg.get("enabled", False):
        logger.info("Curating training data for concision and image-text alignment")
        annotations = curate_annotations(
            annotations=annotations,
            image_dir=image_dir,
            tokenizer=tokenizer,
            keep_top_fraction=curation_cfg.get("keep_top_fraction", 0.8),
            min_similarity=curation_cfg.get("min_similarity"),
            max_tokens_per_word=curation_cfg.get("max_tokens_per_word"),
            clip_model=curation_cfg.get("clip_model"),
        )
        logger.info("Retained %d examples after curation", len(annotations))

    train_ann, val_ann = split_annotations(
        annotations,
        val_size=config["data"].get("val_size", 0.1),
        seed=seed,
    )

    transform = build_transforms(config["data"].get("transforms", ["resize", "normalize"]), config["data"]["image_size"])
    train_loader = build_dataloader(train_ann, image_dir, tokenizer, transform, config, stage_key, shuffle=True)
    val_loader = build_dataloader(val_ann, image_dir, tokenizer, transform, config, stage_key, shuffle=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage_cfg["learning_rate"],
        weight_decay=stage_cfg.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * stage_cfg["epochs"] // stage_cfg.get("gradient_accumulation_steps", 1)
    warmup_steps = int(total_steps * stage_cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = stage_cfg.get("mixed_precision") in {"fp16", "bf16"}
    scaler = GradScaler() if use_amp else None

    writer = SummaryWriter(log_dir=output_dir / "runs")
    best_val_loss = float("inf")
    best_checkpoint = output_dir / "best_model.pt"

    for epoch in range(1, stage_cfg["epochs"] + 1):
        logger.info("Epoch %d/%d", epoch, stage_cfg["epochs"])
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
            max_grad_norm=stage_cfg.get("max_grad_norm", 1.0),
            log_every=config.get("logging", {}).get("log_every", 10),
            gradient_accumulation_steps=stage_cfg.get("gradient_accumulation_steps", 1),
        )
        logger.info("Train metrics: %s", train_metrics)

        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            logger.info("Validation metrics: %s", val_metrics)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": config,
                    },
                    best_checkpoint,
                )
                logger.info("Saved best checkpoint to %s", best_checkpoint)

        if epoch % stage_cfg.get("save_every", 1000) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    return best_checkpoint


def main() -> None:
    """CLI entry point for staged training."""
    parser = argparse.ArgumentParser(description="Train multimodal VLM")
    parser.add_argument("--config", required=True, help="Path to stage config YAML")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True, help="Training stage")
    parser.add_argument("--resume_from", help="Checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train_stage(
        config=config,
        stage_key=args.stage,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
