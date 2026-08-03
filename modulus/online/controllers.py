"""Small, functional online controllers for GCL outer-loop control.

The functions are deliberately low dimensional. They are intended to control
learning-rate multipliers, residual scales, temperatures, operator mixtures, or
agent-routing weights around a stable base system—not to replace the base
optimizer in the first deployment.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class EuclideanOMDState(NamedTuple):
    anchor: Array
    action: Array
    step: Array


class HedgeState(NamedTuple):
    log_weights: Array
    probabilities: Array
    cumulative_losses: Array
    step: Array


class Exp3State(NamedTuple):
    log_weights: Array
    probabilities: Array
    step: Array


def _as_bound(value: float | Array, reference: Array) -> Array:
    return jnp.asarray(value, dtype=reference.dtype)


def _masked_softmax(logits: Array, availability: Array | None) -> Array:
    if availability is None:
        availability = jnp.ones_like(logits, dtype=bool)
    availability = jnp.asarray(availability, dtype=bool)
    if availability.shape != logits.shape:
        raise ValueError("availability must have the same shape as logits")
    if not bool(jnp.any(availability)):
        raise ValueError("at least one action must be available")
    masked = jnp.where(availability, logits, -jnp.inf)
    shifted = masked - jnp.max(masked)
    weights = jnp.where(availability, jnp.exp(shifted), 0.0)
    return weights / jnp.sum(weights)


def euclidean_omd_init(initial_action: Array) -> EuclideanOMDState:
    action = jnp.asarray(initial_action)
    if not jnp.issubdtype(action.dtype, jnp.floating):
        action = action.astype(jnp.float32)
    return EuclideanOMDState(
        anchor=action,
        action=action,
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def optimistic_euclidean_omd_action(
    state: EuclideanOMDState,
    hint: Array,
    learning_rate: float | Array,
    lower: float | Array,
    upper: float | Array,
) -> EuclideanOMDState:
    """Play the optimistic Euclidean mirror step from the current anchor."""

    hint = jnp.asarray(hint, dtype=state.anchor.dtype)
    if hint.shape != state.anchor.shape:
        raise ValueError("hint must have the same shape as the action")
    eta = _as_bound(learning_rate, state.anchor)
    lower_arr = _as_bound(lower, state.anchor)
    upper_arr = _as_bound(upper, state.anchor)
    if bool(jnp.any(lower_arr >= upper_arr)):
        raise ValueError("lower must be strictly below upper")
    action = jnp.clip(state.anchor - eta * hint, lower_arr, upper_arr)
    return state._replace(action=action)


def optimistic_euclidean_omd_update(
    state: EuclideanOMDState,
    gradient: Array,
    learning_rate: float | Array,
    lower: float | Array,
    upper: float | Array,
) -> EuclideanOMDState:
    """Apply observed gradient feedback to the mirror anchor."""

    gradient = jnp.asarray(gradient, dtype=state.anchor.dtype)
    if gradient.shape != state.anchor.shape:
        raise ValueError("gradient must have the same shape as the action")
    eta = _as_bound(learning_rate, state.anchor)
    lower_arr = _as_bound(lower, state.anchor)
    upper_arr = _as_bound(upper, state.anchor)
    anchor = jnp.clip(state.anchor - eta * gradient, lower_arr, upper_arr)
    return EuclideanOMDState(anchor=anchor, action=state.action, step=state.step + 1)


def hedge_init(num_experts: int, prior: Array | None = None) -> HedgeState:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if prior is None:
        probabilities = jnp.full((num_experts,), 1.0 / num_experts)
    else:
        probabilities = jnp.asarray(prior, dtype=jnp.float32)
        if probabilities.shape != (num_experts,):
            raise ValueError("prior must have shape (num_experts,)")
        if bool(jnp.any(probabilities < 0)) or float(jnp.sum(probabilities)) <= 0:
            raise ValueError("prior must be non-negative with positive mass")
        probabilities = probabilities / jnp.sum(probabilities)
    return HedgeState(
        log_weights=jnp.log(jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny)),
        probabilities=probabilities,
        cumulative_losses=jnp.zeros_like(probabilities),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def sleeping_hedge_action(
    state: HedgeState,
    availability: Array | None = None,
) -> HedgeState:
    probabilities = _masked_softmax(state.log_weights, availability)
    return state._replace(probabilities=probabilities)


def sleeping_hedge_update(
    state: HedgeState,
    losses: Array,
    learning_rate: float | Array,
    availability: Array | None = None,
) -> HedgeState:
    losses = jnp.asarray(losses, dtype=state.log_weights.dtype)
    if losses.shape != state.log_weights.shape:
        raise ValueError("losses must have shape (num_experts,)")
    if availability is None:
        availability = jnp.ones_like(losses, dtype=bool)
    availability = jnp.asarray(availability, dtype=bool)
    eta = jnp.asarray(learning_rate, dtype=state.log_weights.dtype)
    probabilities = _masked_softmax(state.log_weights, availability)
    learner_loss = jnp.vdot(probabilities, jnp.where(availability, losses, 0.0))
    # Specialist reduction: an unavailable expert receives the learner loss,
    # preserving relative weight while it sleeps.
    effective_losses = jnp.where(availability, losses, learner_loss)
    log_weights = state.log_weights - eta * effective_losses
    cumulative = state.cumulative_losses + effective_losses
    probabilities = _masked_softmax(log_weights, availability)
    return HedgeState(log_weights, probabilities, cumulative, state.step + 1)


def fixed_share_hedge_update(
    state: HedgeState,
    losses: Array,
    learning_rate: float | Array,
    share: float,
) -> HedgeState:
    """Hedge update with fixed-share mixing for switching comparators."""

    if not 0.0 <= share <= 1.0:
        raise ValueError("share must lie in [0, 1]")
    losses = jnp.asarray(losses, dtype=state.log_weights.dtype)
    if losses.shape != state.log_weights.shape:
        raise ValueError("losses must have shape (num_experts,)")
    eta = jnp.asarray(learning_rate, dtype=state.log_weights.dtype)
    posterior = _masked_softmax(state.log_weights - eta * losses, None)
    uniform = jnp.full_like(posterior, 1.0 / posterior.shape[0])
    probabilities = (1.0 - share) * posterior + share * uniform
    tiny = jnp.finfo(probabilities.dtype).tiny
    log_weights = jnp.log(jnp.maximum(probabilities, tiny))
    cumulative = state.cumulative_losses + losses
    return HedgeState(log_weights, probabilities, cumulative, state.step + 1)


def exp3_init(num_arms: int, prior: Array | None = None) -> Exp3State:
    hedge_state = hedge_init(num_arms, prior)
    return Exp3State(
        log_weights=hedge_state.log_weights,
        probabilities=hedge_state.probabilities,
        step=hedge_state.step,
    )


def sleeping_exp3_distribution(
    state: Exp3State,
    exploration: float,
    availability: Array | None = None,
) -> Exp3State:
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must lie in [0, 1]")
    base = _masked_softmax(state.log_weights, availability)
    if availability is None:
        available = jnp.ones_like(base, dtype=bool)
    else:
        available = jnp.asarray(availability, dtype=bool)
    uniform = available.astype(base.dtype) / jnp.sum(available)
    probabilities = (1.0 - exploration) * base + exploration * uniform
    return state._replace(probabilities=probabilities)


def sleeping_exp3_update(
    state: Exp3State,
    chosen_arm: int,
    observed_loss: float | Array,
    learning_rate: float | Array,
    availability: Array | None = None,
    probability_floor: float = 1e-12,
) -> Exp3State:
    num_arms = state.log_weights.shape[0]
    if not 0 <= chosen_arm < num_arms:
        raise ValueError("chosen_arm is out of range")
    if availability is not None and not bool(jnp.asarray(availability)[chosen_arm]):
        raise ValueError("chosen_arm must be available")
    probability = jnp.maximum(state.probabilities[chosen_arm], probability_floor)
    estimated_loss = jnp.asarray(observed_loss, dtype=state.log_weights.dtype) / probability
    estimate = jnp.zeros_like(state.log_weights).at[chosen_arm].set(estimated_loss)
    eta = jnp.asarray(learning_rate, dtype=state.log_weights.dtype)
    log_weights = state.log_weights - eta * estimate
    return Exp3State(log_weights, state.probabilities, state.step + 1)
