# Multimodal Vision-Language Training

A production-ready PyTorch framework for training small-to-medium multimodal vision-language models with staged freezing, outlier analysis, and diagnostic tooling.

## Overview

This project trains text-and-image models using a staged freezing schedule: the vision encoder is held frozen while the projection connector and text decoder train, then the connector is frozen to tune the remaining stage. The pipeline includes tools to analyse outliers and false positives in model outputs to separate data problems from training problems.

## Features

- **Modular VLM Architecture**: Separate vision encoder, projection connector, and causal text decoder with clean interfaces.
- **Staged Freezing**: YAML-defined stage configs freeze/unfreeze components per training phase.
- **Dataset Pipeline**: Image + text datasets with configurable transforms and tokenization.
- **Data-Centric Curation**: Optional CLIP-based image-text alignment filtering and caption-concision scoring to train on a smaller, higher-quality subset (lower Cost-of-Pass).
- **Training Loop**: Native PyTorch training loop with mixed precision (`torch.cuda.amp`), gradient clipping, EMA, and checkpointing.
- **Diagnostics**: TensorBoard logging of losses, gradients, and learning rates; outlier detection and false-positive analysis.
- **Reproducibility**: Fixed seeds, pinned dependencies, Docker support, and experiment tracking.
- **Testing**: `pytest` suite with unit, integration, and smoke tests.

## Project Structure

```
multimodal-vision-language-training/
├── README.md                      # This file
├── requirements.txt               # Pinned Python dependencies
├── environment.yml                # Conda environment
├── pyproject.toml                 # Build & tooling config
├── Dockerfile                     # Training container
├── docker-compose.yml             # Service orchestration
├── Makefile                       # Common tasks
├── .env.example                   # Environment variables
├── .gitignore                     # Git ignore rules
├── configs/                       # Stage configs
│   ├── train_stage1.yaml
│   ├── train_stage2.yaml
│   └── eval_config.yaml
├── src/                           # Source code
│   ├── __init__.py
│   ├── data.py                    # VLM dataset & transforms
│   ├── data_curation.py           # Caption concision & image-text alignment
│   ├── models.py                  # Vision encoder, connector, text decoder
│   ├── train.py                   # Staged training loop
│   ├── evaluation.py              # Standard metrics (BLEU, CIDEr, accuracy)
│   ├── analysis.py                # Outlier & false-positive analysis
│   ├── utils.py                   # Reproducibility & helpers
│   └── logging_utils.py           # Structured logging
├── tests/                         # Pytest suite
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_data_curation.py
│   ├── test_models.py
│   └── test_analysis.py
├── scripts/                       # Bash entry points
│   ├── setup_env.sh
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── run_eval.sh
│   └── run_tests.sh
└── notebooks/                     # Analysis notebooks
    └── error_analysis.ipynb
```

## Quick Start

### 1. Set Up Environment

```bash
cd programs/multimodal-vision-language-training

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate vlm-training
```

### 2. Prepare Data

Place image files under `data/images/` and captions/annotations in `data/annotations.json` (COCO-like format). The dataset loader expects:

```json
[
  {"image_id": "0001", "image_path": "data/images/0001.jpg", "caption": "A cat sitting on a mat."},
  ...
]
```

For a quick smoke test, generate dummy data:

```bash
python -m src.data --generate_dummy --output_dir data/dummy
```

### 3. Train Stage 1 (Frozen Vision Encoder)

```bash
bash scripts/run_stage1.sh configs/train_stage1.yaml
```

### 4. Train Stage 2 (Tune Remaining Stage)

```bash
bash scripts/run_stage2.sh configs/train_stage2.yaml --resume_from outputs/stage1/best_model.pt
```

### 5. Evaluate & Analyse

```bash
bash scripts/run_eval.sh outputs/stage2/best_model.pt configs/eval_config.yaml
```

Open `notebooks/error_analysis.ipynb` to visualise outliers and false positives.

## Data-Centric Curation for Efficient VLM Training

Recent work (e.g. DatologyAI, 2026) shows that **training on concise, well-aligned data** can dramatically reduce inference Cost-of-Pass (FLOPs per correct answer) while preserving accuracy. This project includes an optional `src/data_curation.py` module that filters VLM training data by:

- **Caption concision**: prefer shorter, information-dense captions over verbose ones.
- **Image-text alignment**: use CLIP similarity to drop mismatched image-caption pairs.

Enable in any stage config:

```yaml
data:
  curation:
    enabled: true
    keep_top_fraction: 0.75      # keep top 75% by combined score
    max_tokens_per_word: 2.5     # drop overly verbose captions
    min_similarity: null         # optional CLIP threshold
    clip_model: "openai/clip-vit-base-patch32"
```

Curation runs once before the train/validation split, so both stages can benefit from the same cleaned data.

## Staged Training

### Stage 1: Vision Encoder Frozen

- **Goal**: Align vision features with the text decoder's embedding space.
- **Trainable**: Projection connector + text decoder.
- **Frozen**: Vision encoder (CLIP/SigLIP or custom CNN).

### Stage 2: Tune Remaining Stage

- **Goal**: Improve captioning/QA quality while preserving vision representations.
- **Trainable**: Text decoder (and optionally projection connector).
- **Frozen**: Vision encoder + projection connector.

Configure each stage via YAML in `configs/train_stage*.yaml`.

## Configuration

Key config sections:

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `vision_encoder`, `text_decoder` | HF model names or local paths |
| `model` | `connector_type` | `linear`, `mlp`, `qformer` |
| `model` | `freeze_vision_encoder` | Freeze vision encoder |
| `stage1` / `stage2` | `epochs`, `lr`, `batch_size`, `freeze_components` | Stage-specific training |
| `data` | `annotations`, `image_dir`, `transforms` | Data paths & augmentations |
| `analysis` | `outlier_threshold`, `fp_iou_threshold` | Diagnostic thresholds |

## Diagnostics

### Outlier Detection

`src/analysis.py` flags samples with:

- Loss value above a percentile threshold.
- Large gradient norm spikes.
- Inconsistent predictions across augmented views.

### False-Positive vs Data-Problem Separation

For each flagged sample, the analysis script computes:

1. **Data quality signals**: OCR errors, duplicate annotations, low inter-annotator overlap.
2. **Model quality signals**: high confidence but wrong answer, repeated tokens, hallucinated objects.

Results are saved to `outputs/analysis/error_report.json` and can be loaded in the notebook.

## Testing

```bash
bash scripts/run_tests.sh
```

Run only fast tests:

```bash
make test-fast
```

## Docker

```bash
# Stage 1
docker compose up --build train-stage1

# Stage 2
docker compose up --build train-stage2

# Evaluation
docker compose up --build eval
```

## Reproducibility Checklist

- [ ] Set `training.seed`.
- [ ] Pin model revisions (`model.revision`).
- [ ] Use Docker or identical environment.
- [ ] Save config with each checkpoint.
- [ ] Log git commit SHA and data version.

## License

Provided as a research/educational template. Respect the licenses of any vision encoders, text decoders, or datasets you use.

## References & Sources

The efficiency and diagnostic ideas in this project are grounded in recent research:

1. **Concision-aware data curation for inference efficiency**  
   *Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation* — DatologyAI (2026).  
   [arXiv:2606.25432](https://arxiv.org/abs/2606.25432)  
   Shows that training VLMs on concise, correct data lowers *Cost-of-Pass* (FLOPs per correct answer) without model compression.

2. **Value-centric data curation**  
   *A Survey on Efficient Large Language Model Training: From Data-centric Perspectives* — ACL 2025.  
   [ACL Anthology](https://aclanthology.org/2025.acl-long.1493.pdf)  
   Surveys data distillation, compression, and value-centric selection for efficient training.

3. **Multimodal vision-language architectures**  
   LLaVA (Liu et al., 2023), Qwen-VL (Bai et al., 2023), and SigLIP (Zhai et al., 2023) established the vision-encoder + projection + text-decoder design pattern used here.

## Acknowledgements

- Hugging Face Transformers
- PyTorch Team
- CLIP / SigLIP / LLaVA research communities
