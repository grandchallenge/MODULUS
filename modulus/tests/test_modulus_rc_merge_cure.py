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


def test_ratified_pending_review_cure_validates() -> None:
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
    broken["retrospective_ratification"]["comment_id"] = 5161244663
    with pytest.raises(ValueError, match="ratification identity drift"):
        validator.validate_record(broken)


def test_ratification_comment_id_is_exact() -> None:
    broken = copy.deepcopy(_record())
    broken["retrospective_ratification"]["comment_id"] = 5161330307
    with pytest.raises(ValueError, match="ratification identity drift"):
        validator.validate_record(broken)


def test_ratification_timestamp_is_exact() -> None:
    broken = copy.deepcopy(_record())
    broken["retrospective_ratification"]["recorded_at"] = "2026-08-03T01:15:52Z"
    with pytest.raises(ValueError, match="ratification identity drift"):
        validator.validate_record(broken)


def test_ratification_must_be_steward_authored() -> None:
    broken = copy.deepcopy(_record())
    broken["retrospective_ratification"]["author"] = "jimsteeg"
    with pytest.raises(ValueError, match="ratification identity drift"):
        validator.validate_record(broken)


def test_pending_review_state_rejects_bound_review() -> None:
    broken = copy.deepcopy(_record())
    broken["independent_cure_review"] = {
        "required": True,
        "review_id": 999,
        "reviewer": "jimsteeg",
        "submitted_at": "2026-08-03T01:20:00Z",
    }
    with pytest.raises(ValueError, match="may not bind review"):
        validator.validate_record(broken)


def test_advanced_state_requires_non_author_review() -> None:
    broken = copy.deepcopy(_record())
    broken["state"] = "reviewed_pending_protected_merge"
    broken["independent_cure_review"] = {
        "required": True,
        "review_id": 999,
        "reviewer": "fyremael",
        "submitted_at": "2026-08-03T01:20:00Z",
    }
    with pytest.raises(ValueError, match="non-author reviewer"):
        validator.validate_record(broken)


def test_author_audit_cannot_be_relabelled_as_non_author() -> None:
    broken = copy.deepcopy(_record())
    broken["pre_merge_evidence"]["formal_reviews_observed"] = [4840145007, 999]
    with pytest.raises(ValueError, match="pre-merge evidence drift"):
        validator.validate_record(broken)


def test_original_placeholder_defect_is_preserved() -> None:
    broken = copy.deepcopy(_record())
    broken["invalid_closeout_comment"]["original_contains_placeholders"] = False
    with pytest.raises(ValueError, match="closeout-comment identity drift"):
        validator.validate_record(broken)


def test_superseded_status_is_fail_closed() -> None:
    broken = copy.deepcopy(_record())
    broken["invalid_closeout_comment"]["currently_explicitly_superseded"] = False
    with pytest.raises(ValueError, match="closeout-comment identity drift"):
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
