"""
Benchmarking module for model evaluation.

Measures accuracy, average latency, and P99 latency on a dataset.
"""

import time
import torch

from quant_pipeline.core.metrics import compute_accuracy


def benchmark(model, tokenizer, texts, labels):
    """
    Benchmark a model on text classification.

    Runs a single forward pass per sample, measures
    latency during the prediction pass itself.

    Parameters
    ----------
    model : nn.Module
        The model to benchmark.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding text.
    texts : list[str]
        Input sentences.
    labels : list[int]
        Ground truth labels.

    Returns
    -------
    dict
        Keys: accuracy, avg_latency_ms, p99_latency_ms
    """
    model.eval()
    preds = []
    latencies = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        preds.append(torch.argmax(outputs.logits, dim=1).item())

    p99_idx = min(int(0.99 * len(latencies)), len(latencies) - 1)

    return {
        "accuracy": compute_accuracy(preds, labels),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p99_latency_ms": sorted(latencies)[p99_idx],
    }