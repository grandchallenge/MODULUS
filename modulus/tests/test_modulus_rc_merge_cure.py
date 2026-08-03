from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_modulus_rc_merge_cure",
    ROOT / "scripts" / "validate_modulus_rc_merge_cure.py",
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _record() -> dict[str, object]:
    return json.loads(
        (ROOT / "governance" / "MODULUS-RC-MERGE-CURE-001.json").read_text(
            encoding="utf-8"
        )
    )


def test_pending_cure_validates() -> None:
    validator.validate()


def test_cannot_invent_pre_merge_approval() -> None:
    broken = copy.deepcopy(_record())
    broken["subject"]["pre_merge_non_author_approval_present"] = True
    with pytest.raises(ValueError, match="subject merge chronology drift"):
        validator.validate_record(broken)


def test_cannot_invent_pre_merge_steward_release() -> None:
    broken = copy.deepcopy(_record())
    broken["subject"]["pre_merge_steward_release_present"] = True
    with pytest.raises(ValueError, match="subject merge chronology drift"):
        validator.validate_record(broken)


def test_placeholder_comment_cannot_be_ratification() -> None:
    broken = copy.deepcopy(_record())
    broken["state"] = "ratified_pending_review"
    broken["retrospective_ratification"] = {
        "required": True,
        "comment_id": 5161244663,
        "author": "jimsteeg",
        "recorded_at": "2026-08-03T01:00:00Z",
    }
    with pytest.raises(ValueError, match="distinct ratification comment"):
        validator.validate_record(broken)


def test_ratification_must_be_steward_authored() -> None:
    broken = copy.deepcopy(_record())
    broken["state"] = "ratified_pending_review"
    broken["retrospective_ratification"] = {
        "required": True,
        "comment_id": 5161249999,
        "author": "jimsteeg",
        "recorded_at": "2026-08-03T01:00:00Z",
    }
    with pytest.raises(ValueError, match="Human Steward-authored"):
        validator.validate_record(broken)


def test_pending_state_rejects_partial_ratification() -> None:
    broken = copy.deepcopy(_record())
    broken["retrospective_ratification"]["comment_id"] = 5161249999
    with pytest.raises(ValueError, match="may not bind a ratification"):
        validator.validate_record(broken)


def test_author_audit_cannot_be_relabelled_as_non_author() -> None:
    broken = copy.deepcopy(_record())
    broken["pre_merge_evidence"]["formal_reviews_observed"] = [4840145007, 999]
    with pytest.raises(ValueError, match="pre-merge evidence drift"):
        validator.validate_record(broken)


def test_boundary_inflation_is_rejected() -> None:
    broken = copy.deepcopy(_record())
    broken["preserved_boundaries"]["modulus_conformant"] = True
    with pytest.raises(ValueError, match="merge-cure boundary drift"):
        validator.validate_record(broken)


def test_false_revert_requirement_is_rejected() -> None:
    broken = copy.deepcopy(_record())
    broken["preserved_boundaries"]["revert_required"] = True
    with pytest.raises(ValueError, match="merge-cure boundary drift"):
        validator.validate_record(broken)
