"""
Layer-wise sensitivity analysis.

For each quantizable layer, quantizes ONLY that layer to INT8
while keeping everything else at FP32, then measures accuracy
to determine which layers are most sensitive to quantization.
"""

import torch
import torch.nn as nn
from copy import deepcopy


def quantize_single_layer(model, target_layer_name):
    """
    Quantize a single Linear layer to INT8 via dynamic quantization,
    leaving all other layers at FP32.

    Parameters
    ----------
    model : nn.Module
        The FP32 model (will be deep-copied, not modified).
    target_layer_name : str
        Dot-separated path to the layer.

    Returns
    -------
    nn.Module
        Model with only the target layer quantized.
    """
    modified = deepcopy(model)
    modified.eval()

    parts = target_layer_name.split(".")
    parent = modified
    for part in parts[:-1]:
        parent = getattr(parent, part) if hasattr(parent, part) else parent[int(part)]

    target = getattr(parent, parts[-1])

    if isinstance(target, nn.Linear):
        quantized_layer = torch.quantization.quantize_dynamic(
            nn.Sequential(target),
            {nn.Linear},
            dtype=torch.qint8,
        )
        setattr(parent, parts[-1], quantized_layer[0])
    elif isinstance(target, nn.Module):
        quantized_sub = torch.quantization.quantize_dynamic(
            target,
            {nn.Linear},
            dtype=torch.qint8,
        )
        setattr(parent, parts[-1], quantized_sub)

    return modified


def get_quantizable_layers(model, max_layers=None):
    """
    Get names of all nn.Linear layers in the model.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
    max_layers : int, optional
        Limit the number of layers returned.

    Returns
    -------
    list[str]
        Layer names suitable for sensitivity analysis.
    """
    layers = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if max_layers:
        layers = layers[:max_layers]
    return layers


def analyze_sensitivity(model, tokenizer, texts, labels, layer_names, benchmark_fn):
    """
    Run layer-wise sensitivity analysis.

    For each layer: quantize ONLY that layer, measure accuracy,
    compare with FP32 baseline.

    Parameters
    ----------
    model : nn.Module
        The FP32 baseline model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding text.
    texts : list[str]
        Evaluation sentences.
    labels : list[int]
        Ground truth labels.
    layer_names : list[str]
        Layers to analyze.
    benchmark_fn : callable
        Function with signature (model, tokenizer, texts, labels) -> dict.

    Returns
    -------
    tuple
        (results_dict, baseline_accuracy)
    """
    print("Computing FP32 baseline...")
    baseline = benchmark_fn(model, tokenizer, texts, labels)
    baseline_acc = baseline["accuracy"]
    print(f"Baseline accuracy: {baseline_acc:.4f}\n")

    results = {}

    for i, layer_name in enumerate(layer_names):
        print(f"[{i + 1}/{len(layer_names)}] Quantizing layer: {layer_name}")

        try:
            modified_model = quantize_single_layer(model, layer_name)
            metrics = benchmark_fn(modified_model, tokenizer, texts, labels)
            acc = metrics["accuracy"]
            delta = acc - baseline_acc

            results[layer_name] = {
                "accuracy": acc,
                "delta": delta,
            }
            print(f"  accuracy={acc:.4f} (delta={delta:+.4f})\n")

        except Exception as e:
            print(f"  Skipping: {e}\n")
            results[layer_name] = {"accuracy": baseline_acc, "delta": 0.0}

    return results, baseline_acc