"""
Visualization for sensitivity analysis and robustness results.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sensitivity(results, baseline_acc, save_path="outputs/sensitivity.png"):
    """
    Plot layer sensitivity as a horizontal bar chart showing
    accuracy delta per layer.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sorted_items = sorted(
        results.items(), key=lambda x: abs(x[1]["delta"]), reverse=True
    )

    layer_names = [name.replace("distilbert.transformer.", "") for name, _ in sorted_items]
    deltas = [item["delta"] for _, item in sorted_items]

    colors = ["#d32f2f" if d < -0.005 else "#388e3c" if d > 0.005 else "#757575" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(4, len(layer_names) * 0.4)))
    ax.barh(layer_names, deltas, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy Delta (vs FP32 baseline)")
    ax.set_title(f"Layer Sensitivity to INT8 Quantization (baseline={baseline_acc:.4f})")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sensitivity plot saved to {save_path}")


def plot_robustness_comparison(robustness_results, save_path="outputs/robustness.png"):
    """
    Plot robustness comparison across modes and perturbations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    modes = list(robustness_results.keys())
    perturbations = list(robustness_results[modes[0]].keys())

    x = range(len(perturbations))
    width = 0.8 / len(modes)

    for i, mode in enumerate(modes):
        values = [robustness_results[mode][p] for p in perturbations]
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness: FP32 vs Quantized Under Distribution Shift")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Robustness plot saved to {save_path}")