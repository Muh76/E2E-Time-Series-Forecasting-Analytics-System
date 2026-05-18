"""
Backward-compatibility shim — augmentation logic lives in data.etl.transformers.augment.

Import from the canonical location instead:
    from data.etl.transformers import augment, augment_timeseries, add_gaussian_noise
"""

from .transformers.augment import add_gaussian_noise, augment, augment_timeseries

__all__ = [
    "add_gaussian_noise",
    "augment",
    "augment_timeseries",
]
