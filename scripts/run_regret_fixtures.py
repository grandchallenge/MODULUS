"""Run deterministic reference fixtures for the GCL Regret Contract Standard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from modulus.online import (
    RegretTracker,
    euclidean_omd_init,
    exp3_init,
    fixed_share_hedge_update,
    hedge_init,
    optimistic_euclidean_omd_action,
    optimistic_euclidean_omd_update,
    sleeping_exp3_distribution,
    sleeping_exp3_update,
    sleeping_hedge_action,
)


def kibo_fixture(rounds: int = 200) -> dict[str, float]:
    ordinary = euclidean_omd_init(jnp.array([0.0]))
    optimistic = euclidean_omd_init(jnp.array([0.0]))
    ordinary_loss = 0.0
    optimistic_loss = 0.0
    previous_gradient = jnp.array([0.0])
    for step in range(rounds):
        gradient = jnp.array([np.sin(step / 12.0)], dtype=jnp.float32)
        ordinary = optimistic_euclidean_omd_action(
            ordinary, jnp.zeros_like(gradient), 0.15, -1.0, 1.0
        )
        optimistic = optimistic_euclidean_omd_action(
            optimistic, previous_gradient, 0.15, -1.0, 1.0
        )
        ordinary_loss += float(jnp.vdot(gradient, ordinary.action))
        optimistic_loss += float(jnp.vdot(gradient, optimistic.action))
        ordinary = optimistic_euclidean_omd_update(
            ordinary, gradient, 0.15, -1.0, 1.0
        )
        optimistic = optimistic_euclidean_omd_update(
            optimistic, gradient, 0.15, -1.0, 1.0
        )
        previous_gradient = gradient
    return {
        "ordinary_linear_loss": ordinary_loss,
        "optimistic_linear_loss": optimistic_loss,
        "optimism_gain": ordinary_loss - optimistic_loss,
    }


def aether_fixture(rounds: int = 300, seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    state = exp3_init(4)
    tracker = RegretTracker(4)
    cumulative = 0.0
    for step in range(rounds):
        availability = np.array(
            [True, step % 5 != 0, step % 7 != 0, step % 11 != 0], dtype=bool
        )
        phase = 0 if step < rounds // 2 else 1
        base_losses = np.array(
            [
                0.15 if phase == 0 else 0.65,
                0.35,
                0.65 if phase == 0 else 0.15,
                0.45,
            ],
            dtype=np.float32,
        )
        state = sleeping_exp3_distribution(
            state, exploration=0.08, availability=jnp.asarray(availability)
        )
        probabilities = np.asarray(state.probabilities)
        chosen = int(rng.choice(len(probabilities), p=probabilities))
        observed = float(base_losses[chosen])
        cumulative += observed
        comparator_losses = np.where(availability, base_losses, observed)
        tracker.update(observed, comparator_losses)
        state = sleeping_exp3_update(
            state,
            chosen_arm=chosen,
            observed_loss=observed,
            learning_rate=0.05,
            availability=jnp.asarray(availability),
        )
    snapshot = tracker.snapshot()
    return {
        "cumulative_loss": cumulative,
        "static_regret": snapshot.static_regret,
        "tracking_regret_k1": tracker.tracking_regret(1),
    }


def spindle_fixture(rounds: int = 200) -> dict[str, float]:
    state = hedge_init(3)
    tracker = RegretTracker(3)
    for step in range(rounds):
        losses = (
            jnp.array([0.1, 0.4, 0.7])
            if step < rounds // 2
            else jnp.array([0.7, 0.4, 0.1])
        )
        state = sleeping_hedge_action(state)
        learner_loss = float(jnp.vdot(state.probabilities, losses))
        tracker.update(learner_loss, [float(value) for value in losses])
        state = fixed_share_hedge_update(
            state, losses, learning_rate=1.5, share=0.005
        )
    snapshot = tracker.snapshot()
    return {
        "cumulative_loss": snapshot.cumulative_loss,
        "static_regret": snapshot.static_regret,
        "tracking_regret_k1": tracker.tracking_regret(1),
        "maximum_interval_static_regret": tracker.maximum_interval_static_regret(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {
        "schema_version": "1.0.0",
        "fixtures": {
            "kibo_optimistic_omd": kibo_fixture(),
            "aether_sleeping_bandit": aether_fixture(),
            "spindle_dynamic_scheduler": spindle_fixture(),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
