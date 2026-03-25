"""
Quantization utilities with cross-platform support.

Applies FP16, INT8 PTQ (dynamic), and INT8 QAT to PyTorch models.
Includes runtime validation and fallback.
"""

import torch
import torch.nn as nn
import platform
from copy import deepcopy

from quant_pipeline.quantization.qat_trainer import train_qat


def get_system_info():
    """Return (is_mac, is_arm) based on platform detection."""
    system = platform.system()
    processor = platform.processor()
    is_mac = system == "Darwin"
    is_arm = "arm" in processor.lower() or "apple" in processor.lower()
    return is_mac, is_arm


def is_apple_silicon():
    """Check if running on Apple Silicon (M1/M2/M3)."""
    is_mac, is_arm = get_system_info()
    return is_mac and is_arm


def _select_backend():
    """Select the best available quantization backend."""
    if is_apple_silicon():
        return "qnnpack"
    else:
        return "fbgemm"


def _validate_quantized_model(model, tokenizer):
    """
    Run a test forward pass to confirm the quantized model works.

    Returns True if inference succeeds, False otherwise.
    """
    try:
        dummy = tokenizer("test", return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            model(**dummy)
        return True
    except Exception:
        return False


def apply_quantization(model, mode, tokenizer=None, train_data=None):
    """
    Apply the requested quantization mode to a model.

    Parameters
    ----------
    model : nn.Module
        The FP32 pretrained model.
    mode : str
        One of 'fp32', 'fp16', 'int8_ptq', 'int8_qat'.
    tokenizer : PreTrainedTokenizer, optional
        Needed for QAT training and model validation.
    train_data : tuple, optional
        (texts, labels) for QAT fine-tuning.

    Returns
    -------
    nn.Module
        The quantized model.
    """
    if mode == "fp32":
        return model

    elif mode == "fp16":
        return model.half()

    elif mode == "int8_ptq":
        return _apply_int8_ptq(model, tokenizer)

    elif mode == "int8_qat":
        return _apply_int8_qat(model, tokenizer, train_data)

    else:
        raise ValueError(f"Invalid quantization mode: '{mode}'")


def _apply_int8_ptq(model, tokenizer=None):
    """Apply INT8 post-training dynamic quantization."""
    try:
        backend = _select_backend()
        torch.backends.quantized.engine = backend

        model_q = torch.quantization.quantize_dynamic(
            deepcopy(model),
            {nn.Linear},
            dtype=torch.qint8,
        )

        if tokenizer and not _validate_quantized_model(model_q, tokenizer):
            raise RuntimeError("INT8 model failed forward pass validation")

        print(f"[PTQ] INT8 dynamic quantization applied (backend={backend})")
        return model_q

    except Exception as e:
        print(f"[PTQ] INT8 failed ({e}) — falling back to FP16")
        return model.half()


def _apply_int8_qat(model, tokenizer=None, train_data=None):
    """Apply INT8 quantization-aware training."""
    if train_data is None:
        print("[QAT] No training data provided — falling back to FP16")
        return model.half()

    try:
        texts, labels = train_data
        backend = _select_backend()
        torch.backends.quantized.engine = backend

        qat_model = deepcopy(model)
        qat_model = train_qat(qat_model, tokenizer, texts, labels, epochs=1)

        qat_model = torch.quantization.quantize_dynamic(
            qat_model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        if tokenizer and not _validate_quantized_model(qat_model, tokenizer):
            raise RuntimeError("QAT model failed forward pass validation")

        print(f"[QAT] INT8 quantization-aware training applied (backend={backend})")
        return qat_model

    except Exception as e:
        print(f"[QAT] Failed ({e}) — falling back to FP16")
        return model.half()