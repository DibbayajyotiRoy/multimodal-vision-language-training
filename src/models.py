"""Vision-language model architecture and component loading."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class MLPConnector(nn.Module):
    """Simple MLP connector mapping vision features to text embedding space."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        """Initialize MLP connector.

        Args:
            in_dim: Vision encoder output dimension.
            out_dim: Text decoder embedding dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to text embedding space.

        Args:
            vision_features: Vision encoder output tensor.

        Returns:
            Projected features.
        """
        x = self.fc1(vision_features)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x)


class LinearConnector(nn.Module):
    """Linear projection connector."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projection(vision_features)


class VisionLanguageModel(nn.Module):
    """End-to-end vision-language model."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_decoder: nn.Module,
        connector: nn.Module,
        freeze_vision_encoder: bool = True,
    ) -> None:
        """Initialize VLM.

        Args:
            vision_encoder: Pre-trained vision encoder.
            text_decoder: Pre-trained causal language model.
            connector: Vision-to-text connector.
            freeze_vision_encoder: Whether to freeze vision encoder weights.
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.connector = connector

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            pixel_values: Input images.
            input_ids: Tokenized text input.
            attention_mask: Attention mask.
            labels: Labels for language modeling loss.

        Returns:
            Dictionary with loss, logits, and vision features.
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        # Pool vision features: mean over spatial tokens if 3D.
        vision_features = vision_outputs.last_hidden_state
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)

        projected = self.connector(vision_features)

        # Prepend projected vision embedding as soft prompt.
        inputs_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([projected.unsqueeze(1), inputs_embeds[:, 1:]], dim=1)

        # Build attention mask to account for prepended token.
        batch_size = input_ids.size(0)
        prompt_mask = torch.ones(
            (batch_size, 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask = torch.cat([prompt_mask, attention_mask[:, 1:]], dim=1)

        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "vision_features": vision_features,
        }

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        tokenizer: Any,
        max_length: int = 128,
        **generation_kwargs: Any,
    ) -> list[str]:
        """Generate captions from images.

        Args:
            pixel_values: Input images.
            tokenizer: Text tokenizer.
            max_length: Max generation length.
            generation_kwargs: Additional generation arguments.

        Returns:
            List of decoded strings.
        """
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state
        if vision_features.dim() == 3:
            vision_features = vision_features.mean(dim=1)
        projected = self.connector(vision_features).unsqueeze(1)

        batch_size = pixel_values.size(0)
        start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=pixel_values.device,
        )

        generated = input_ids
        for _ in range(max_length):
            inputs_embeds = self.text_decoder.get_input_embeddings()(generated)
            inputs_embeds = torch.cat([projected, inputs_embeds], dim=1)
            attention_mask = torch.ones_like(inputs_embeds[:, :, 0])
            outputs = self.text_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token == tokenizer.eos_token_id):
                break

        return tokenizer.batch_decode(generated, skip_special_tokens=True)


def load_components(config: dict[str, Any], device: torch.device | None = None) -> tuple[VisionLanguageModel, Any]:
    """Load vision encoder, text decoder, tokenizer, and assemble VLM.

    Args:
        config: Model configuration dictionary.
        device: Target device.

    Returns:
        Tuple of (VisionLanguageModel, tokenizer).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision_encoder = AutoModel.from_pretrained(
        config["vision_encoder"],
        revision=config.get("vision_encoder_revision", "main"),
        trust_remote_code=True,
    )
    text_decoder = AutoModelForCausalLM.from_pretrained(
        config["text_decoder"],
        revision=config.get("text_decoder_revision", "main"),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_decoder"],
        revision=config.get("text_decoder_revision", "main"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_dim = vision_encoder.config.hidden_size
    text_dim = text_decoder.config.hidden_size

    connector_type = config.get("connector_type", "linear")
    if connector_type == "mlp":
        connector = MLPConnector(
            in_dim=vision_dim,
            out_dim=text_dim,
            hidden_dim=config.get("connector_hidden_dim", text_dim),
            dropout=config.get("dropout", 0.1),
        )
    else:
        connector = LinearConnector(in_dim=vision_dim, out_dim=text_dim)

    model = VisionLanguageModel(
        vision_encoder=vision_encoder,
        text_decoder=text_decoder,
        connector=connector,
        freeze_vision_encoder=config.get("freeze_vision_encoder", True),
    )
    model.to(device)
    return model, tokenizer


def set_component_freezing(model: VisionLanguageModel, freeze_components: list[str]) -> None:
    """Freeze specified components by name.

    Args:
        model: VLM instance.
        freeze_components: List of component names ('vision_encoder', 'connector', 'text_decoder').
    """
    for name in freeze_components:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"Unknown component: {name}")
        for param in module.parameters():
            param.requires_grad = False


def set_component_training(model: VisionLanguageModel, trainable_components: list[str]) -> None:
    """Ensure specified components are trainable.

    Args:
        model: VLM instance.
        trainable_components: List of component names.
    """
    for name in trainable_components:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"Unknown component: {name}")
        for param in module.parameters():
            param.requires_grad = True
