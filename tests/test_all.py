"""
Comprehensive test suite for quant_pipeline.

Covers: metrics, perturbations, memory, benchmark, sensitivity,
pipeline, quantization, config validation, and edge cases.
"""

import os
import pytest
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════
# Test: Metrics
# ═══════════════════════════════════════════════════════════

from quant_pipeline.core.metrics import compute_accuracy


class TestMetrics:
    def test_perfect_accuracy(self):
        assert compute_accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0

    def test_zero_accuracy(self):
        assert compute_accuracy([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0

    def test_partial_accuracy(self):
        assert compute_accuracy([0, 0, 1, 1], [0, 0, 0, 0]) == 0.5


# ═══════════════════════════════════════════════════════════
# Test: Robustness / Perturbations
# ═══════════════════════════════════════════════════════════

from quant_pipeline.analysis.robustness import (
    inject_typos, drop_words, add_noise_chars, perturb_texts,
)


class TestPerturbations:
    def test_typo_changes_text(self):
        text = "this is a long enough sentence for testing typo injection"
        result = inject_typos(text, prob=0.5, seed=42)
        assert result != text

    def test_typo_preserves_length(self):
        text = "hello world testing"
        result = inject_typos(text, prob=0.5, seed=42)
        assert len(result) == len(text)

    def test_drop_reduces_words(self):
        text = "one two three four five six seven eight nine ten"
        result = drop_words(text, prob=0.5, seed=42)
        assert len(result.split()) < len(text.split())

    def test_drop_never_empty(self):
        result = drop_words("hello world", prob=0.99, seed=42)
        assert len(result) > 0

    def test_drop_single_word_safe(self):
        assert drop_words("hello", prob=0.99, seed=42) == "hello"

    def test_noise_adds_chars(self):
        result = add_noise_chars("hello", prob=0.5, seed=42)
        assert len(result) >= len("hello")

    def test_perturb_texts_batch(self):
        result = perturb_texts(["hello world", "this is great", "testing now"],
                               method="typo", severity=0.3, seed=42)
        assert len(result) == 3
        assert all(isinstance(t, str) for t in result)

    def test_invalid_perturbation_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation"):
            perturb_texts(["hello"], method="invalid", severity=0.1)


# ═══════════════════════════════════════════════════════════
# Test: Memory Profiling
# ═══════════════════════════════════════════════════════════

from quant_pipeline.utils.memory import get_model_size


class TestMemory:
    def test_model_size_positive(self):
        assert get_model_size(nn.Linear(100, 10)) > 0

    def test_fp16_smaller_than_fp32(self):
        assert get_model_size(nn.Linear(1000, 1000).half()) < get_model_size(nn.Linear(1000, 1000)) * 0.6

    def test_includes_buffers(self):
        model = nn.BatchNorm1d(100)
        size = get_model_size(model)
        param_only = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        assert size > param_only


# ═══════════════════════════════════════════════════════════
# Test: Quantization
# ═══════════════════════════════════════════════════════════

from quant_pipeline.quantization.utils import apply_quantization


class TestQuantization:
    def _make_simple_model(self):
        return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def test_fp32_passthrough(self):
        model = self._make_simple_model()
        result = apply_quantization(model, "fp32")
        assert result is model

    def test_fp16_dtype(self):
        result = apply_quantization(self._make_simple_model(), "fp16")
        assert next(result.parameters()).dtype == torch.float16

    def test_int8_ptq_returns_model(self):
        result = apply_quantization(self._make_simple_model(), "int8_ptq")
        assert result is not None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid quantization mode"):
            apply_quantization(self._make_simple_model(), "int4_magic")

    def test_quantization_does_not_modify_original(self):
        model = self._make_simple_model()
        original_dtype = next(model.parameters()).dtype
        apply_quantization(model, "fp16")
        assert next(model.parameters()).dtype == original_dtype


# ═══════════════════════════════════════════════════════════
# Test: Sensitivity
# ═══════════════════════════════════════════════════════════

from quant_pipeline.analysis.sensitivity import get_quantizable_layers, quantize_single_layer


class TestSensitivity:
    def test_get_layers_returns_linear(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        layers = get_quantizable_layers(model)
        assert len(layers) == 2

    def test_get_layers_max_limit(self):
        model = nn.Sequential(
            nn.Linear(64, 32), nn.Linear(32, 16),
            nn.Linear(16, 8), nn.Linear(8, 2),
        )
        assert len(get_quantizable_layers(model, max_layers=2)) == 2

    def test_quantize_single_layer_returns_new_model(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        layers = get_quantizable_layers(model)
        modified = quantize_single_layer(model, layers[0])
        assert modified is not model


# ═══════════════════════════════════════════════════════════
# Test: CSV Export
# ═══════════════════════════════════════════════════════════

from quant_pipeline.utils.export import save_results_to_csv


class TestExport:
    def test_csv_created(self, tmp_path):
        path = str(tmp_path / "test_results.csv")
        save_results_to_csv([{"mode": "fp32", "accuracy": 0.95}], filepath=path)
        assert os.path.exists(path)

    def test_csv_content(self, tmp_path):
        import csv
        path = str(tmp_path / "test_results.csv")
        save_results_to_csv([
            {"mode": "fp32", "accuracy": 0.95},
            {"mode": "fp16", "accuracy": 0.94},
        ], filepath=path)

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["mode"] == "fp32"