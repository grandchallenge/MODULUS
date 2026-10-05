#!/usr/bin/env python3
"""Run the deterministic PS-DUST estimator sanity fixture.

This is Gate 0 only.  It checks estimator semantics against JAX autodiff on a
controlled nonlinear objective.  It is not evidence that structured directions
improve DUST or transformer training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from modulus.estimators.structured_zeroth_order import (
    cosine_similarity,
    estimate_gradient,
    relative_l2_error,
    sample_directions,
)

SCHEMES = ("gaussian", "rademacher", "hadamard")


def nonlinear_fixture(point: jax.Array) -> jax.Array:
    indices = jnp.arange(point.size, dtype=point.dtype) + 1.0
    diagonal = 0.5 + indices / point.size
    quadratic = 0.5 * jnp.sum(diagonal * point * point)
    oscillatory = 0.075 * jnp.sum(jnp.sin(1.7 * point + 0.013 * indices))
    quartic = 0.005 * jnp.sum(point**4)
    return quadratic + oscillatory + quartic


def run_fixture(*, dimension: int, populations: list[int], seeds: int, sigma: float) -> dict:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("dimension must be a positive power of two")
    if any(population <= 0 or population > dimension for population in populations):
        raise ValueError("populations must be positive and no larger than dimension")
    if seeds <= 0:
        raise ValueError("seeds must be positive")

    point = jnp.linspace(-0.9, 0.7, dimension, dtype=jnp.float32)
    reference = jax.grad(nonlinear_fixture)(point)
    rows = []

    for scheme in SCHEMES:
        for antithetic in (False, True):
            for population in populations:
                cosines = []
                errors = []
                evaluations = None
                for seed in range(seeds):
                    directions = sample_directions(
                        jax.random.PRNGKey(1000 + seed),
                        dimension=dimension,
                        population=population,
                        scheme=scheme,
                    )
                    estimate = estimate_gradient(
                        nonlinear_fixture,
                        point,
                        directions,
                        sigma=sigma,
                        antithetic=antithetic,
                    )
                    cosines.append(float(cosine_similarity(estimate.gradient, reference)))
                    errors.append(float(relative_l2_error(estimate.gradient, reference)))
                    evaluations = estimate.evaluations

                rows.append(
                    {
                        "scheme": scheme,
                        "antithetic": antithetic,
                        "population": population,
                        "forward_evaluations": evaluations,
                        "cosine_mean": sum(cosines) / len(cosines),
                        "cosine_min": min(cosines),
                        "relative_l2_mean": sum(errors) / len(errors),
                        "relative_l2_max": max(errors),
                    }
                )

    return {
        "schema_version": "1.0.0",
        "experiment_id": "PS-DUST-GATE0-001",
        "status": "DETERMINISTIC_SANITY_FIXTURE_ONLY",
        "claim_boundary": (
            "Validates estimator implementation on one controlled nonlinear objective only; "
            "does not establish DUST, transformer, training, scaling, or novelty claims."
        ),
        "dimension": dimension,
        "populations": populations,
        "seeds": seeds,
        "sigma": sigma,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--populations", default="8,16,32,64")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    populations = [int(value) for value in args.populations.split(",") if value]
    result = run_fixture(
        dimension=args.dimension,
        populations=populations,
        seeds=args.seeds,
        sigma=args.sigma,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
