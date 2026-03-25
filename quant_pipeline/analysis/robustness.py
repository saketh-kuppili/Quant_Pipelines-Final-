"""
Robustness evaluation under distribution shift.

Applies controlled text perturbations and compares how FP32
vs quantized models degrade under noisy inputs.
"""

import random
import string


def inject_typos(text, prob=0.08, seed=None):
    """Swap adjacent characters with given probability."""
    if seed is not None:
        random.seed(seed)

    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < prob:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def drop_words(text, prob=0.1, seed=None):
    """Randomly drop words from a sentence."""
    if seed is not None:
        random.seed(seed)

    words = text.split()
    if len(words) <= 1:
        return text
    kept = [w for w in words if random.random() > prob]
    return " ".join(kept) if kept else words[0]


def add_noise_chars(text, prob=0.05, seed=None):
    """Insert random characters into the text."""
    if seed is not None:
        random.seed(seed)

    result = []
    for ch in text:
        result.append(ch)
        if random.random() < prob:
            result.append(random.choice(string.ascii_lowercase))
    return "".join(result)


def perturb_texts(texts, method="typo", severity=0.08, seed=42):
    """
    Apply a perturbation to a list of texts.

    Parameters
    ----------
    texts : list[str]
        Original sentences.
    method : str
        'typo', 'drop', or 'noise'.
    severity : float
        Perturbation probability.
    seed : int
        Random seed.

    Returns
    -------
    list[str]
        Perturbed sentences.
    """
    fn_map = {
        "typo": inject_typos,
        "drop": drop_words,
        "noise": add_noise_chars,
    }

    if method not in fn_map:
        raise ValueError(f"Unknown perturbation '{method}'. Use: {list(fn_map.keys())}")

    fn = fn_map[method]
    return [fn(t, prob=severity, seed=seed + i) for i, t in enumerate(texts)]


def evaluate_robustness(model, tokenizer, texts, labels, benchmark_fn, perturbations=None):
    """
    Evaluate model robustness under multiple perturbation types.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    tokenizer : PreTrainedTokenizer
        Tokenizer.
    texts : list[str]
        Clean test sentences.
    labels : list[int]
        Ground truth labels.
    benchmark_fn : callable
        (model, tokenizer, texts, labels) -> dict with 'accuracy'.
    perturbations : list[dict], optional
        List of {"method": str, "severity": float} dicts.

    Returns
    -------
    dict
        Mapping from perturbation name to accuracy.
    """
    if perturbations is None:
        perturbations = [
            {"method": "typo", "severity": 0.1},
            {"method": "drop", "severity": 0.15},
            {"method": "noise", "severity": 0.08},
        ]

    clean_metrics = benchmark_fn(model, tokenizer, texts, labels)
    results = {"clean": clean_metrics["accuracy"]}

    for perturb in perturbations:
        method = perturb["method"]
        severity = perturb["severity"]
        key = f"{method}_{severity}"

        perturbed = perturb_texts(texts, method=method, severity=severity)
        metrics = benchmark_fn(model, tokenizer, perturbed, labels)
        results[key] = metrics["accuracy"]

        drop = clean_metrics["accuracy"] - metrics["accuracy"]
        print(f"  {key}: accuracy={metrics['accuracy']:.4f} (drop={drop:+.4f})")

    return results