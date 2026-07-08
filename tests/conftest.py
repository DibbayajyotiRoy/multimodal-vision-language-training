"""Pytest fixtures and shared helpers."""

from __future__ import annotations

import pytest
import torch

from src.models import LinearConnector, MLPConnector, VisionLanguageModel


@pytest.fixture
def dummy_vision_encoder() -> torch.nn.Module:
    """Return a tiny vision encoder stand-in."""

    class DummyVisionEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 64})()
            self.fc = torch.nn.Linear(3 * 32 * 32, 64)

        def forward(self, pixel_values: torch.Tensor) -> type:
            b = pixel_values.size(0)
            x = pixel_values.view(b, -1)
            hidden = self.fc(x).unsqueeze(1)
            Output = type("Output", (), {"last_hidden_state": hidden})
            return Output()

    return DummyVisionEncoder()


@pytest.fixture
def dummy_text_decoder() -> torch.nn.Module:
    """Return a tiny causal LM stand-in."""

    class DummyTextDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 32})()
            self.embed = torch.nn.Embedding(100, 32)
            self.lm_head = torch.nn.Linear(32, 100)

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embed

        def forward(
            self,
            inputs_embeds: torch.Tensor | None = None,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            return_dict: bool = True,
        ) -> type:
            if inputs_embeds is None:
                inputs_embeds = self.embed(input_ids)
            logits = self.lm_head(inputs_embeds)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
            Output = type("Output", (), {"loss": loss, "logits": logits})
            return Output()

    return DummyTextDecoder()


@pytest.fixture
def dummy_vlm(dummy_vision_encoder, dummy_text_decoder) -> VisionLanguageModel:
    """Return a VLM with dummy components."""
    connector = LinearConnector(in_dim=64, out_dim=32)
    return VisionLanguageModel(
        vision_encoder=dummy_vision_encoder,
        text_decoder=dummy_text_decoder,
        connector=connector,
        freeze_vision_encoder=True,
    )


@pytest.fixture
def sample_config(tmp_path) -> dict:
    """Return a minimal config for tests."""
    return {
        "model": {
            "vision_encoder": "dummy",
            "text_decoder": "dummy",
            "connector_type": "linear",
            "connector_hidden_dim": 64,
            "dropout": 0.1,
            "freeze_vision_encoder": True,
            "max_length": 16,
        },
        "stage1": {
            "output_dir": str(tmp_path / "stage1"),
            "run_name": "test-stage1",
            "freeze_components": ["vision_encoder"],
            "trainable_components": ["connector", "text_decoder"],
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "mixed_precision": null,
            "save_every": 1000,
            "eval_every": 1000,
            "seed": 42,
        },
        "stage2": {
            "output_dir": str(tmp_path / "stage2"),
            "run_name": "test-stage2",
            "freeze_components": ["vision_encoder", "connector"],
            "trainable_components": ["text_decoder"],
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "mixed_precision": null,
            "save_every": 1000,
            "eval_every": 1000,
            "seed": 42,
        },
        "data": {
            "annotations": str(tmp_path / "annotations.json"),
            "image_dir": str(tmp_path / "images"),
            "image_size": 32,
            "val_size": 0.2,
            "num_workers": 0,
            "transforms": ["resize", "normalize"],
        },
        "analysis": {
            "outlier_loss_percentile": 90,
            "fp_iou_threshold": 0.3,
            "min_confidence": 0.8,
        },
        "logging": {"log_every": 1, "report_to": "none"},
    }
