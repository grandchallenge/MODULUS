from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "modulus_rc_merge_cure.schema.json"
RECORD_PATH = ROOT / "governance" / "MODULUS-RC-MERGE-CURE-001.json"
DOCUMENT_PATH = ROOT / "docs" / "governance" / "MODULUS-RC-MERGE-CURE-001.md"

EXPECTED_SUBJECT = {
    "repository": "fyremael/MODULUS",
    "pull_request": 1,
    "reviewed_head": "a78ed738d7d4b9de03d0212a54d8ec35fd15ce4d",
    "merge_commit": "959829113cca27a4f14d42ab620b78c9a890f1bf",
    "merged_at": "2026-08-03T00:56:42Z",
    "pre_merge_non_author_approval_present": False,
    "pre_merge_steward_release_present": False,
}
EXPECTED_AUTHORITY = {
    "repository": "grandchallenge/gcl-standards",
    "commit": "4bb7e09cbd8ddac521447cb1386bc501f9ac5b12",
}
EXPECTED_EVIDENCE = {
    "ci_run": 30774910406,
    "delegated_audit_review": 4840145007,
    "formal_reviews_observed": [4840145007],
}
EXPECTED_INVALID_COMMENT = {
    "repository": "grandchallenge/.github",
    "issue": 4,
    "comment_id": 5161244663,
    "author": "jimsteeg",
    "contains_placeholders": True,
    "human_steward_authorship_valid": False,
    "classification": "superseded_placeholder_attestation",
}
EXPECTED_BOUNDARIES = {
    "standard_status": "candidate",
    "modulus_conformant": False,
    "implementation_status": "reference_candidate",
    "programme_adoption_complete": False,
    "revert_required": False,
    "controller_optimality_claimed": False,
    "deployment_safety_claimed": False,
    "product_or_commercial_claimed": False,
}


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_schema(schema: dict[str, object]) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("merge-cure schema draft drift")
    if schema.get("$id") != (
        "https://grandchallenge.ai/schemas/modulus-rc-merge-cure-1.0.0.json"
    ):
        raise ValueError("merge-cure schema identity drift")
    if schema.get("additionalProperties") is not False:
        raise ValueError("merge-cure schema must remain closed")
    required = schema.get("required")
    if not isinstance(required, list) or "$schema" not in required:
        raise ValueError("merge-cure schema must require its schema pointer")


def validate_pending_fields(record: dict[str, object]) -> None:
    ratification = record.get("retrospective_ratification")
    if not isinstance(ratification, dict):
        raise ValueError("retrospective_ratification must be an object")
    review = record.get("independent_cure_review")
    if not isinstance(review, dict):
        raise ValueError("independent_cure_review must be an object")

    state = record.get("state")
    if state == "ratification_pending":
        if ratification != {
            "required": True,
            "comment_id": None,
            "author": None,
            "recorded_at": None,
        }:
            raise ValueError("pending cure may not bind a ratification identity")
        if review != {
            "required": True,
            "review_id": None,
            "reviewer": None,
            "submitted_at": None,
        }:
            raise ValueError("pending cure may not bind an independent review")
        return

    if ratification.get("comment_id") in (None, 5161244663):
        raise ValueError("advanced cure requires a distinct ratification comment")
    if ratification.get("author") != "fyremael":
        raise ValueError("ratification must be Human Steward-authored")


def validate_record(record: dict[str, object]) -> None:
    if record.get("$schema") != "../schemas/modulus_rc_merge_cure.schema.json":
        raise ValueError("merge-cure schema pointer drift")
    if record.get("schema_version") != "1.0.0":
        raise ValueError("merge-cure schema version drift")
    if record.get("record_id") != "MODULUS-RC-MERGE-CURE-001":
        raise ValueError("merge-cure record identity drift")
    if record.get("state") not in {
        "ratification_pending",
        "ratified_pending_review",
        "reviewed_pending_protected_merge",
        "protected_complete",
    }:
        raise ValueError("unknown merge-cure state")
    if record.get("subject") != EXPECTED_SUBJECT:
        raise ValueError("subject merge chronology drift")
    if record.get("canonical_authority") != EXPECTED_AUTHORITY:
        raise ValueError("canonical authority drift")
    if record.get("pre_merge_evidence") != EXPECTED_EVIDENCE:
        raise ValueError("pre-merge evidence drift")
    if record.get("invalid_closeout_comment") != EXPECTED_INVALID_COMMENT:
        raise ValueError("invalid closeout-comment identity drift")
    if record.get("preserved_boundaries") != EXPECTED_BOUNDARIES:
        raise ValueError("merge-cure boundary drift")

    validate_pending_fields(record)

    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    for required in (
        "neither:",
        "an independently attributable `APPROVED` review",
        "a Human Steward release naming exact head",
        "comment `5161244663`",
        "superseded placeholder draft",
        "No content revert is presently required",
        "`GCL-RC-00` remains candidate",
    ):
        if required not in document:
            raise ValueError(f"missing merge-cure boundary: {required}")


def validate() -> None:
    validate_schema(load_json(SCHEMA_PATH))
    validate_record(load_json(RECORD_PATH))


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"MODULUS merge-cure validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("MODULUS merge-cure validation passed")
