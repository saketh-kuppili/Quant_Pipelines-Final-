"""
quant_pipeline — Layer-Aware Quantization Pipeline for DistilBERT.

Provides FP32, FP16, INT8 PTQ, and INT8 QAT inference modes
with layer-wise sensitivity analysis and robustness evaluation.
"""

__all__ = ["Pipeline"]


def __getattr__(name):
    if name == "Pipeline":
        from quant_pipeline.core.pipeline import Pipeline
        return Pipeline
    raise AttributeError(f"module 'quant_pipeline' has no attribute {name!r}")