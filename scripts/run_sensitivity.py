"""
Sensitivity analysis script.

Quantizes each layer individually to INT8 and measures accuracy
impact compared to the FP32 baseline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.data.loaders import load_sst2
from quant_pipeline.core.benchmark import benchmark
from quant_pipeline.analysis.sensitivity import analyze_sensitivity, get_quantizable_layers
from quant_pipeline.analysis.visualization import plot_sensitivity


def run():
    print("Starting sensitivity analysis...\n")

    model = load_model()
    tokenizer = load_tokenizer()

    texts, labels = load_sst2(sample_size=50)
    print(f"Loaded {len(texts)} samples\n")

    layers = get_quantizable_layers(model, max_layers=10)
    print(f"Analyzing {len(layers)} layers...\n")

    results, baseline_acc = analyze_sensitivity(
        model, tokenizer, texts, labels, layers, benchmark
    )

    print(f"\n{'='*50}")
    print("  SENSITIVITY SUMMARY")
    print(f"{'='*50}")
    print(f"  Baseline (FP32): {baseline_acc:.4f}\n")

    sorted_results = sorted(
        results.items(), key=lambda x: abs(x[1]["delta"]), reverse=True
    )
    for name, data in sorted_results:
        print(f"  {name}")
        print(f"    accuracy={data['accuracy']:.4f}  delta={data['delta']:+.4f}")

    plot_sensitivity(results, baseline_acc)

    print("\nDone! Saved to outputs/sensitivity.png")


if __name__ == "__main__":
    run()