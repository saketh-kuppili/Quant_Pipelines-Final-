"""
SST-2 dataset loading from HuggingFace.

Returns (texts, labels) tuples for evaluation and training.
"""

from datasets import load_dataset


def load_sst2(split="validation", sample_size=200):
    """
    Load SST-2 dataset samples.

    Parameters
    ----------
    split : str
        Dataset split ('train' or 'validation').
    sample_size : int
        Number of samples to load.

    Returns
    -------
    tuple
        (texts: list[str], labels: list[int])
    """
    try:
        dataset = load_dataset("glue", "sst2", split=split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SST-2 dataset. Check internet connection. Error: {e}"
        )

    sample_size = min(sample_size, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(sample_size))

    return dataset["sentence"], dataset["label"]