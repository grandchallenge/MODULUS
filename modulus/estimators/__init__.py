"""Gradient-estimation utilities used by MODULUS research experiments."""

from .structured_zeroth_order import (
    DirectionScheme,
    GradientEstimate,
    cosine_similarity,
    estimate_gradient,
    hadamard_matrix,
    relative_l2_error,
    sample_directions,
)

__all__ = [
    "DirectionScheme",
    "GradientEstimate",
    "cosine_similarity",
    "estimate_gradient",
    "hadamard_matrix",
    "relative_l2_error",
    "sample_directions",
]
