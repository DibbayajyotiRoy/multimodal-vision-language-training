"""Tests for analysis utilities."""

from __future__ import annotations

from src.analysis import (
    analyze_false_positives,
    detect_outliers,
    detect_repeated_tokens,
    lexical_overlap,
)


def test_detect_outliers() -> None:
    """Outlier detection should flag highest-loss records."""
    records = [{"image_id": i, "loss": float(i)} for i in range(100)]
    outliers = detect_outliers(records, percentile=95)
    assert len(outliers) == 5
    assert outliers[0]["loss"] == 99.0


def test_detect_repeated_tokens() -> None:
    """Detect repetitive generation."""
    assert detect_repeated_tokens("hello hello hello hello", min_repeats=4)
    assert not detect_repeated_tokens("hello world", min_repeats=4)


def test_lexical_overlap() -> None:
    """Lexical overlap should range 0-1."""
    assert lexical_overlap("a b c", "a b c") == 1.0
    assert lexical_overlap("a b c", "d e f") == 0.0
    assert 0.0 < lexical_overlap("a b c", "a d e") < 1.0


def test_analyze_false_positives() -> None:
    """Analysis should split model vs data problems."""
    predictions = [
        {"image_id": "1", "prediction": "a b c", "reference": "a b c", "confidence": 0.9},
        {"image_id": "2", "prediction": "x y z", "reference": "a b c", "confidence": 0.9},
        {"image_id": "3", "prediction": "a b", "reference": "x y z", "confidence": 0.3},
    ]
    report = analyze_false_positives(predictions, iou_threshold=0.3, min_confidence=0.8)
    assert report["model_problems_count"] == 1
    assert report["data_problems_count"] == 1
    assert report["avg_iou"] > 0
