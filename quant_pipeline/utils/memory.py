"""
Model memory profiling.

Includes both parameters and buffers (quantized models store
scale/zero-point as buffers).
"""


def get_model_size(model):
    """
    Compute model memory footprint in megabytes.

    Parameters
    ----------
    model : nn.Module
        The model to measure.

    Returns
    -------
    float
        Total size in MB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_mb = (param_bytes + buffer_bytes) / (1024 ** 2)
    return round(total_mb, 2)