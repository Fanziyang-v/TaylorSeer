"""Elementwise operations on tensor or flat tuple features."""
from collections.abc import Callable
import torch

Feature = torch.Tensor | tuple[torch.Tensor, ...]


def map_feature(fn: Callable, *features: Feature) -> Feature:
    if isinstance(features[0], tuple):
        return tuple(fn(*items) for items in zip(*features, strict=True))
    return fn(*features)
