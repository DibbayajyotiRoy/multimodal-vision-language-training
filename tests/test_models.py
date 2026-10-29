"""Tests for model architecture."""

from __future__ import annotations

import torch

from src.models import MLPConnector, set_component_freezing
from src.utils import count_trainable_parameters


def test_mlp_connector() -> None:
    """MLP connector should project feature dimensions."""
    connector = MLPConnector(in_dim=16, out_dim=32, hidden_dim=64)
    x = torch.randn(2, 16)
    out = connector(x)
    assert out.shape == (2, 32)


def test_vision_language_model_forward(dummy_vlm) -> None:
    """VLM forward pass should return loss and logits."""
    batch_size, seq_len = 2, 10
    pixel_values = torch.randn(batch_size, 3, 32, 32)
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    outputs = dummy_vlm(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert "loss" in outputs
    assert "logits" in outputs
    assert outputs["logits"].shape[:2] == (batch_size, seq_len)


def test_freezing(dummy_vlm) -> None:
    """Setting component freezing should change requires_grad."""
    set_component_freezing(dummy_vlm, ["connector", "text_decoder"])
    trainable, total = count_trainable_parameters(dummy_vlm)
    assert trainable == 0


def test_vision_encoder_frozen_by_default(dummy_vlm) -> None:
    """Vision encoder should be frozen when freeze_vision_encoder=True."""
    for param in dummy_vlm.vision_encoder.parameters():
        assert not param.requires_grad
