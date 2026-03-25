"""
Main benchmark script.

Evaluates FP32, FP16, INT8 PTQ, and INT8 QAT on SST-2, then runs
robustness evaluation comparing FP32 vs INT8 under perturbations.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_pipeline.models.nlp.distilbert import load_model, load_tokenizer
from quant_pipeline.quantization.utils import apply_quantization
from quant_pipeline.data.loaders import load_sst2
from quant_pipeline.core.benchmark import benchmark
from quant_pipeline.utils.memory import get_model_size
from quant_pipeline.utils.export import save_results_to_csv, save_robustness_to_csv
from quant_pipeline.analysis.robustness import evaluate_robustness
from quant_pipeline.analysis.visualization import plot_robustness_comparison


def run():
    texts, labels = load_sst2(split="validation", sample_size=200)
    tokenizer = load_tokenizer()

    modes = ["fp32", "fp16", "int8_ptq", "int8_qat"]
    results = []
    models_for_robustness = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  {mode.upper()}")
        print(f"{'='*50}")

        model = load_model()

        train_data = None
        if mode == "int8_qat":
            train_data = load_sst2(split="train", sample_size=300)

        model = apply_quantization(
            model, mode, tokenizer=tokenizer, train_data=train_data
        )

        metrics = benchmark(model, tokenizer, texts, labels)
        mem = get_model_size(model)

        result = {
            "mode": mode,
            "accuracy": round(metrics["accuracy"], 4),
            "avg_latency_ms": round(metrics["avg_latency_ms"], 2),
            "p99_latency_ms": round(metrics["p99_latency_ms"], 2),
            "memory_mb": mem,
        }
        results.append(result)

        if mode in ("fp32", "int8_ptq"):
            models_for_robustness[mode] = model

        print(f"  Mode:      {mode}")
        print(f"  Accuracy:  {result['accuracy']}")
        print(f"  Latency:   {result['avg_latency_ms']}ms (avg), {result['p99_latency_ms']}ms (p99)")
        print(f"  Memory:    {result['memory_mb']} MB")

    save_results_to_csv(results)

    print(f"\n{'='*50}")
    print("  ROBUSTNESS EVALUATION")
    print(f"{'='*50}")

    robustness_texts, robustness_labels = load_sst2(split="validation", sample_size=100)
    robustness_all = {}

    for mode_name, model in models_for_robustness.items():
        print(f"\n  {mode_name.upper()}:")
        rob = evaluate_robustness(
            model, tokenizer, robustness_texts, robustness_labels, benchmark
        )
        robustness_all[mode_name] = rob

    if robustness_all:
        save_robustness_to_csv(robustness_all)
        plot_robustness_comparison(robustness_all)

    print("\nDone!")


if __name__ == "__main__":
    run()