import json

import pytest

from modulus.online import (
    ActionSpace,
    ComparatorClass,
    FeedbackModel,
    GeometrySpec,
    GuaranteeType,
    LossSpec,
    RegretContract,
)


def _contract() -> RegretContract:
    return RegretContract(
        contract_id="kibo.lr-group.v1",
        controller="optimistic_euclidean_omd",
        action_semantics="groupwise learning-rate multipliers",
        feedback=FeedbackModel.FULL_INFORMATION,
        comparator=ComparatorClass.PATH_LENGTH_BOUNDED,
        guarantee=GuaranteeType.DYNAMIC,
        geometry=GeometrySpec(
            action_space=ActionSpace.BOX,
            norm="l2",
            regularizer="half_squared_l2",
            projection_or_retraction="clip_[0.25,4.0]",
        ),
        loss=LossSpec(
            name="progress_stability_surrogate",
            lower_bound=0.0,
            upper_bound=1.0,
            components=("progress", "boundary_risk", "compute"),
        ),
        optimism_hint_source="koopman_boundary_predictor",
        telemetry=("cumulative_loss", "action", "dynamic_regret"),
    )


def test_contract_round_trip() -> None:
    contract = _contract()
    encoded = contract.to_json()
    decoded = RegretContract.from_dict(json.loads(encoded))
    assert decoded == contract


def test_dynamic_requires_moving_comparator() -> None:
    contract = _contract()
    invalid = RegretContract(
        **{**contract.__dict__, "comparator": ComparatorClass.FIXED}
    )
    with pytest.raises(ValueError, match="moving comparator"):
        invalid.validate()


def test_delayed_feedback_requires_delay_bound() -> None:
    contract = _contract()
    invalid = RegretContract(
        **{**contract.__dict__, "feedback": FeedbackModel.DELAYED_BANDIT}
    )
    with pytest.raises(ValueError, match="delayed_feedback_max_steps"):
        invalid.validate()
