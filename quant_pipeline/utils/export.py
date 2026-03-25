"""
CSV export for benchmark and robustness results.
"""

import csv
import os


def save_results_to_csv(results, filepath="outputs/results.csv"):
    """
    Save benchmark results to CSV.

    Parameters
    ----------
    results : list[dict]
        Each dict should have keys like requested_mode, actual_mode,
        accuracy, avg_latency_ms, p99_latency_ms, memory_mb.
    filepath : str
        Output path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filepath}")


def save_robustness_to_csv(robustness_results, filepath="outputs/robustness.csv"):
    """
    Save robustness comparison results to CSV.

    Parameters
    ----------
    robustness_results : dict
        {mode: {perturbation: accuracy, ...}, ...}
    filepath : str
        Output path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for mode, perturbations in robustness_results.items():
        for perturb_name, accuracy in perturbations.items():
            rows.append({
                "mode": mode,
                "perturbation": perturb_name,
                "accuracy": round(accuracy, 4),
            })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "perturbation", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Robustness results saved to {filepath}")