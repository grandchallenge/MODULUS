from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from modulus.estimators.structured_zeroth_order import (
    cosine_similarity,
    estimate_gradient,
    hadamard_matrix,
    relative_l2_error,
    sample_directions,
)


def test_hadamard_matrix_is_orthogonal() -> None:
    matrix = hadamard_matrix(8)
    gram = matrix @ matrix.T
    assert jnp.allclose(gram, 8.0 * jnp.eye(8), atol=1e-6)


def test_hadamard_requires_power_of_two_dimension() -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        hadamard_matrix(6)


def test_randomized_hadamard_full_basis_recovers_linear_gradient() -> None:
    dimension = 16
    key = jax.random.PRNGKey(17)
    target = jnp.linspace(-1.0, 1.0, dimension)
    point = jnp.linspace(0.2, 0.8, dimension)
    directions = sample_directions(
        key,
        dimension=dimension,
        population=dimension,
        scheme="hadamard",
    )

    estimate = estimate_gradient(
        lambda x: jnp.vdot(target, x),
        point,
        directions,
        sigma=0.05,
        antithetic=True,
    )

    assert estimate.evaluations == 2 * dimension
    assert jnp.allclose(estimate.gradient, target, atol=2e-5, rtol=2e-5)


def test_metric_helpers_identify_exact_match() -> None:
    reference = jnp.asarray([1.0, -2.0, 3.0])
    assert float(cosine_similarity(reference, reference)) == pytest.approx(1.0, abs=1e-7)
    assert float(relative_l2_error(reference, reference)) == pytest.approx(0.0, abs=1e-7)


def test_one_sided_evaluation_count_includes_clean_baseline() -> None:
    key = jax.random.PRNGKey(5)
    point = jnp.ones((4,))
    directions = sample_directions(key, dimension=4, population=3, scheme="rademacher")
    estimate = estimate_gradient(
        lambda x: jnp.sum(x * x),
        point,
        directions,
        sigma=0.01,
        antithetic=False,
    )
    assert estimate.evaluations == 4
