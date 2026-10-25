"""Structured zeroth-order gradient estimators for bounded research experiments.

The module is intentionally model-agnostic.  It provides perturbation families that
can be injected at an activation site and a common forward-only estimator that can
be scored against an exact autodiff oracle during controlled experiments.

This code does not claim that any perturbation family improves DUST or transformer
training.  That is an empirical question for the PS-DUST campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp

Array = jax.Array
DirectionScheme = Literal["gaussian", "rademacher", "hadamard"]


@dataclass(frozen=True)
class GradientEstimate:
    """One forward-only gradient estimate and its forward-evaluation count."""

    gradient: Array
    evaluations: int


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def hadamard_matrix(dimension: int, *, dtype=jnp.float32) -> Array:
    """Return the Sylvester Hadamard matrix of power-of-two order ``dimension``."""
    if not _is_power_of_two(dimension):
        raise ValueError("Hadamard directions require a positive power-of-two dimension.")

    matrix = jnp.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < dimension:
        top = jnp.concatenate((matrix, matrix), axis=1)
        bottom = jnp.concatenate((matrix, -matrix), axis=1)
        matrix = jnp.concatenate((top, bottom), axis=0)
    return matrix


def sample_directions(
    key: Array,
    *,
    dimension: int,
    population: int,
    scheme: DirectionScheme,
    dtype=jnp.float32,
) -> Array:
    """Draw perturbation directions with coordinate covariance equal to identity in expectation.

    Gaussian and Rademacher directions have independent coordinates.  Hadamard directions
    are sampled without replacement from a randomized orthogonal basis.  A Hadamard block
    therefore requires ``population <= dimension`` and a power-of-two dimension.
    """
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if population <= 0:
        raise ValueError("population must be positive")

    if scheme == "gaussian":
        return jax.random.normal(key, (population, dimension), dtype=dtype)

    if scheme == "rademacher":
        bits = jax.random.bernoulli(key, 0.5, (population, dimension))
        return jnp.where(bits, jnp.asarray(1.0, dtype=dtype), jnp.asarray(-1.0, dtype=dtype))

    if scheme == "hadamard":
        if population > dimension:
            raise ValueError("Hadamard population cannot exceed dimension within one basis block.")
        matrix = hadamard_matrix(dimension, dtype=dtype)
        sign_key, row_key = jax.random.split(key)
        signs = jnp.where(
            jax.random.bernoulli(sign_key, 0.5, (dimension,)),
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(-1.0, dtype=dtype),
        )
        randomized = matrix * signs[None, :]
        rows = jax.random.permutation(row_key, dimension)[:population]
        return randomized[rows]

    raise ValueError(f"unknown direction scheme: {scheme}")


def estimate_gradient(
    loss_fn: Callable[[Array], Array],
    point: Array,
    directions: Array,
    *,
    sigma: float,
    antithetic: bool,
) -> GradientEstimate:
    """Estimate ``grad(loss_fn)(point)`` from forward evaluations only.

    The directions are assumed to have identity coordinate covariance in expectation.
    With antithetic sampling, each direction uses a centered two-sided finite difference.
    Otherwise a clean baseline plus one perturbed evaluation per direction is used.
    """
    point = jnp.asarray(point)
    directions = jnp.asarray(directions, dtype=point.dtype)
    if directions.ndim != 2 or directions.shape[1] != point.size:
        raise ValueError("directions must have shape [population, point.size]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    flat = point.reshape(-1)
    step = jnp.asarray(sigma, dtype=flat.dtype)

    def evaluate(delta: Array) -> Array:
        value = loss_fn((flat + delta).reshape(point.shape))
        return jnp.asarray(value, dtype=jnp.float32)

    if antithetic:
        plus = jax.vmap(lambda direction: evaluate(step * direction))(directions)
        minus = jax.vmap(lambda direction: evaluate(-step * direction))(directions)
        directional = (plus - minus) / (2.0 * step)
        evaluations = 2 * int(directions.shape[0])
    else:
        base = jnp.asarray(loss_fn(point), dtype=jnp.float32)
        plus = jax.vmap(lambda direction: evaluate(step * direction))(directions)
        directional = (plus - base) / step
        evaluations = int(directions.shape[0]) + 1

    gradient = jnp.mean(directional[:, None] * directions, axis=0)
    return GradientEstimate(gradient=gradient.reshape(point.shape), evaluations=evaluations)


def cosine_similarity(estimate: Array, reference: Array, *, eps: float = 1e-12) -> Array:
    """Cosine similarity between flattened arrays."""
    estimate = jnp.ravel(estimate).astype(jnp.float32)
    reference = jnp.ravel(reference).astype(jnp.float32)
    denom = jnp.linalg.norm(estimate) * jnp.linalg.norm(reference)
    return jnp.dot(estimate, reference) / jnp.maximum(denom, eps)


def relative_l2_error(estimate: Array, reference: Array, *, eps: float = 1e-12) -> Array:
    """Relative L2 error against a reference gradient."""
    estimate = jnp.ravel(estimate).astype(jnp.float32)
    reference = jnp.ravel(reference).astype(jnp.float32)
    return jnp.linalg.norm(estimate - reference) / jnp.maximum(jnp.linalg.norm(reference), eps)
