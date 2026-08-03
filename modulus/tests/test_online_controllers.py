import jax.numpy as jnp
import pytest

from modulus.online import (
    euclidean_omd_init,
    exp3_init,
    fixed_share_hedge_update,
    hedge_init,
    optimistic_euclidean_omd_action,
    optimistic_euclidean_omd_update,
    sleeping_exp3_distribution,
    sleeping_exp3_update,
    sleeping_hedge_action,
    sleeping_hedge_update,
)


def test_optimistic_omd_respects_box() -> None:
    state = euclidean_omd_init(jnp.array([0.0, 0.5]))
    state = optimistic_euclidean_omd_action(
        state,
        hint=jnp.array([-10.0, 10.0]),
        learning_rate=1.0,
        lower=-1.0,
        upper=1.0,
    )
    assert jnp.allclose(state.action, jnp.array([1.0, -1.0]))
    state = optimistic_euclidean_omd_update(
        state,
        gradient=jnp.array([10.0, -10.0]),
        learning_rate=1.0,
        lower=-1.0,
        upper=1.0,
    )
    assert jnp.allclose(state.anchor, jnp.array([-1.0, 1.0]))


def test_hedge_downweights_high_loss_expert() -> None:
    state = hedge_init(3)
    for _ in range(10):
        state = sleeping_hedge_update(
            state,
            losses=jnp.array([0.0, 0.5, 1.0]),
            learning_rate=0.5,
        )
    assert state.probabilities[0] > state.probabilities[1] > state.probabilities[2]


def test_sleeping_hedge_masks_unavailable_expert() -> None:
    state = hedge_init(3)
    state = sleeping_hedge_action(state, jnp.array([True, False, True]))
    assert state.probabilities[1] == 0.0
    assert jnp.isclose(jnp.sum(state.probabilities), 1.0)


def test_sleeping_exp3_updates_only_chosen_arm() -> None:
    availability = jnp.array([True, False, True])
    state = exp3_init(3)
    state = sleeping_exp3_distribution(state, exploration=0.1, availability=availability)
    before = state.log_weights
    state = sleeping_exp3_update(
        state,
        chosen_arm=2,
        observed_loss=0.5,
        learning_rate=0.2,
        availability=availability,
    )
    assert state.log_weights[0] == before[0]
    assert state.log_weights[1] == before[1]
    assert state.log_weights[2] < before[2]


def test_exp3_rejects_sleeping_choice() -> None:
    state = exp3_init(2)
    state = sleeping_exp3_distribution(
        state, exploration=0.1, availability=jnp.array([True, False])
    )
    with pytest.raises(ValueError, match="available"):
        sleeping_exp3_update(
            state,
            chosen_arm=1,
            observed_loss=0.5,
            learning_rate=0.2,
            availability=jnp.array([True, False]),
        )


def test_fixed_share_recovers_after_switch() -> None:
    standard = hedge_init(2)
    tracking = hedge_init(2)
    standard_loss = 0.0
    tracking_loss = 0.0
    for step in range(100):
        losses = jnp.array([0.0, 1.0]) if step < 50 else jnp.array([1.0, 0.0])
        standard = sleeping_hedge_action(standard)
        tracking = sleeping_hedge_action(tracking)
        standard_loss += float(jnp.vdot(standard.probabilities, losses))
        tracking_loss += float(jnp.vdot(tracking.probabilities, losses))
        standard = sleeping_hedge_update(standard, losses, learning_rate=0.3)
        tracking = fixed_share_hedge_update(
            tracking, losses, learning_rate=0.3, share=0.04
        )
    assert tracking_loss < standard_loss
