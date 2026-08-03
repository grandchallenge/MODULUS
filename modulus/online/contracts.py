"""Typed contracts for auditable online-control components.

The contract is intentionally independent of a particular optimizer. It records
what an adaptive controller is allowed to do, what feedback it observes, which
comparator class defines success, and which geometry converts feedback into an
action.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionSpace(str, Enum):
    BOX = "box"
    SIMPLEX = "simplex"
    FINITE_EXPERTS = "finite_experts"
    MANIFOLD = "manifold"
    CUSTOM = "custom"


class FeedbackModel(str, Enum):
    FULL_INFORMATION = "full_information"
    BANDIT = "bandit"
    DELAYED_FULL_INFORMATION = "delayed_full_information"
    DELAYED_BANDIT = "delayed_bandit"


class ComparatorClass(str, Enum):
    FIXED = "fixed"
    PATH_LENGTH_BOUNDED = "path_length_bounded"
    K_SWITCH = "k_switch"
    INTERVAL_FIXED = "interval_fixed"


class GuaranteeType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    TRACKING = "tracking"
    STRONGLY_ADAPTIVE = "strongly_adaptive"


@dataclass(frozen=True)
class GeometrySpec:
    action_space: ActionSpace
    norm: str
    regularizer: str
    projection_or_retraction: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeometrySpec":
        return cls(
            action_space=ActionSpace(value["action_space"]),
            norm=str(value["norm"]),
            regularizer=str(value["regularizer"]),
            projection_or_retraction=value.get("projection_or_retraction"),
        )


@dataclass(frozen=True)
class LossSpec:
    name: str
    lower_bound: float
    upper_bound: float
    components: tuple[str, ...] = ()
    convex_surrogate: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("loss.name must be non-empty")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("loss.upper_bound must exceed loss.lower_bound")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LossSpec":
        return cls(
            name=str(value["name"]),
            lower_bound=float(value["lower_bound"]),
            upper_bound=float(value["upper_bound"]),
            components=tuple(str(item) for item in value.get("components", ())),
            convex_surrogate=bool(value.get("convex_surrogate", True)),
        )


@dataclass(frozen=True)
class RegretContract:
    contract_id: str
    controller: str
    action_semantics: str
    feedback: FeedbackModel
    comparator: ComparatorClass
    guarantee: GuaranteeType
    geometry: GeometrySpec
    loss: LossSpec
    delayed_feedback_max_steps: int | None = None
    optimism_hint_source: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    telemetry: tuple[str, ...] = (
        "cumulative_loss",
        "static_regret",
        "action",
        "feedback_delay",
    )
    schema_version: str = "1.0.0"

    def validate(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("contract_id must be non-empty")
        if not self.controller.strip():
            raise ValueError("controller must be non-empty")
        if not self.action_semantics.strip():
            raise ValueError("action_semantics must be non-empty")
        self.loss.validate()

        delayed = self.feedback in {
            FeedbackModel.DELAYED_FULL_INFORMATION,
            FeedbackModel.DELAYED_BANDIT,
        }
        if delayed and self.delayed_feedback_max_steps is None:
            raise ValueError("delayed feedback requires delayed_feedback_max_steps")
        if self.delayed_feedback_max_steps is not None and self.delayed_feedback_max_steps < 0:
            raise ValueError("delayed_feedback_max_steps must be non-negative")

        if self.guarantee is GuaranteeType.DYNAMIC and self.comparator not in {
            ComparatorClass.PATH_LENGTH_BOUNDED,
            ComparatorClass.K_SWITCH,
        }:
            raise ValueError("dynamic guarantees require a moving comparator class")
        if (
            self.guarantee is GuaranteeType.TRACKING
            and self.comparator is not ComparatorClass.K_SWITCH
        ):
            raise ValueError("tracking guarantees require comparator=k_switch")
        if (
            self.guarantee is GuaranteeType.STRONGLY_ADAPTIVE
            and self.comparator is not ComparatorClass.INTERVAL_FIXED
        ):
            raise ValueError(
                "strongly_adaptive guarantees require comparator=interval_fixed"
            )

        required = {"cumulative_loss", "action"}
        missing = required.difference(self.telemetry)
        if missing:
            raise ValueError(f"telemetry is missing required fields: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feedback"] = self.feedback.value
        value["comparator"] = self.comparator.value
        value["guarantee"] = self.guarantee.value
        value["geometry"]["action_space"] = self.geometry.action_space.value
        value["loss"]["components"] = list(self.loss.components)
        value["telemetry"] = list(self.telemetry)
        value["constraints"] = dict(self.constraints)
        return value

    def to_json(self, *, indent: int = 2) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegretContract":
        contract = cls(
            contract_id=str(value["contract_id"]),
            controller=str(value["controller"]),
            action_semantics=str(value["action_semantics"]),
            feedback=FeedbackModel(value["feedback"]),
            comparator=ComparatorClass(value["comparator"]),
            guarantee=GuaranteeType(value["guarantee"]),
            geometry=GeometrySpec.from_dict(value["geometry"]),
            loss=LossSpec.from_dict(value["loss"]),
            delayed_feedback_max_steps=value.get("delayed_feedback_max_steps"),
            optimism_hint_source=value.get("optimism_hint_source"),
            constraints=dict(value.get("constraints", {})),
            telemetry=tuple(str(item) for item in value.get("telemetry", ())),
            schema_version=str(value.get("schema_version", "1.0.0")),
        )
        contract.validate()
        return contract
