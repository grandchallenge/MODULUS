from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = (
    ROOT
    / "governance"
    / "settings-readback"
    / "evidence"
    / "GCL-GHOS-MODULUS-OWNER-CONTROLS-001.receipt.json"
)
DIGEST = RECEIPT.with_suffix(RECEIPT.suffix + ".sha256")
SCHEMA = (
    ROOT
    / "governance"
    / "settings-readback"
    / "GCL-GHOS-MODULUS-OWNER-CONTROLS-001.receipt.schema.json"
)
EXPECTED_SHA256 = "bb08be8289fadc455ee413ec59c12a76f22d5cf1021669708c9a88be86f782ba"


def load_receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def test_receipt_digest_and_canonical_bytes() -> None:
    raw = RECEIPT.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    assert raw == canonical_bytes(payload)
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert DIGEST.read_text(encoding="utf-8") == f"{EXPECTED_SHA256}  {RECEIPT.name}\n"


def test_schema_is_closed_and_binds_operation() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["operation_id"]["const"] == (
        "GCL-GHOS-MODULUS-OWNER-CONTROLS-001"
    )
    assert schema["properties"]["repository"]["const"] == "grandchallenge/MODULUS"
    assert schema["properties"]["api_version"]["const"] == "2026-03-10"


def test_authority_and_actor_are_exact() -> None:
    payload = load_receipt()
    assert payload["actor"] == {
        "id": 17925951,
        "login": "fyremael",
        "repository_admin": True,
    }
    assert payload["authority"] == {
        "campaign_issue": "grandchallenge/gcl-standards#22",
        "main_ruleset_id": 20266757,
        "parent_issue": "grandchallenge/MODULUS#10",
        "profile_merge": "c027ceafe1a2226ce8abeec36c739aeaa45ec784",
        "remediation_issue": "grandchallenge/MODULUS#6",
        "surfaces_merge": "7f9efc5a818655454e6117baa596e8382221a874",
    }
    started = datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(payload["completed_at"].replace("Z", "+00:00"))
    assert completed >= started


def test_mutation_sequence_is_closed() -> None:
    payload = load_receipt()
    observed = [
        (item["method"], item["path"], item["status"])
        for item in payload["mutations"]
    ]
    assert observed == [
        ("PATCH", "/repos/grandchallenge/MODULUS", 200),
        ("PUT", "/repos/grandchallenge/MODULUS/rulesets/20266757", 200),
        ("POST", "/repos/grandchallenge/MODULUS/rulesets", 201),
        ("PUT", "/repos/grandchallenge/MODULUS/vulnerability-alerts", 204),
        ("PUT", "/repos/grandchallenge/MODULUS/automated-security-fixes", 204),
        ("PUT", "/repos/grandchallenge/MODULUS/private-vulnerability-reporting", 204),
        ("PATCH", "/repos/grandchallenge/MODULUS/code-scanning/default-setup", 202),
    ]


def test_repository_merge_settings_match_authorization() -> None:
    repository = load_receipt()["post_state"]["repository"]["payload"]
    assert {
        "allow_auto_merge": repository["allow_auto_merge"],
        "allow_merge_commit": repository["allow_merge_commit"],
        "allow_rebase_merge": repository["allow_rebase_merge"],
        "allow_squash_merge": repository["allow_squash_merge"],
        "allow_update_branch": repository["allow_update_branch"],
        "delete_branch_on_merge": repository["delete_branch_on_merge"],
    } == {
        "allow_auto_merge": True,
        "allow_merge_commit": True,
        "allow_rebase_merge": False,
        "allow_squash_merge": True,
        "allow_update_branch": False,
        "delete_branch_on_merge": True,
    }


def test_main_ruleset_is_exact() -> None:
    ruleset = load_receipt()["post_state"]["main_ruleset"]["payload"]
    assert ruleset["id"] == 20266757
    assert ruleset["name"] == "GCL protected main"
    assert ruleset["target"] == "branch"
    assert ruleset["enforcement"] == "active"
    assert ruleset["bypass_actors"] == []
    assert ruleset["conditions"] == {
        "ref_name": {"exclude": [], "include": ["~DEFAULT_BRANCH"]}
    }
    assert [rule["type"] for rule in ruleset["rules"]] == [
        "deletion",
        "non_fast_forward",
        "pull_request",
        "required_status_checks",
    ]
    pull_request = next(
        rule["parameters"] for rule in ruleset["rules"] if rule["type"] == "pull_request"
    )
    assert pull_request == {
        "allowed_merge_methods": ["merge", "squash"],
        "dismiss_stale_reviews_on_push": True,
        "dismissal_restriction": {"allowed_actors": [], "enabled": False},
        "require_code_owner_review": False,
        "require_last_push_approval": False,
        "required_approving_review_count": 0,
        "required_review_thread_resolution": True,
        "required_reviewers": [],
    }
    status = next(
        rule["parameters"]
        for rule in ruleset["rules"]
        if rule["type"] == "required_status_checks"
    )
    assert status == {
        "do_not_enforce_on_create": False,
        "required_status_checks": [
            {"context": "test-and-lint (3.10)"},
            {"context": "test-and-lint (3.11)"},
            {"context": "test-and-lint (3.12)"},
            {"context": "benchmark-report"},
            {"context": "policy / policy"},
            {"context": "security / action-policy"},
        ],
        "strict_required_status_checks_policy": True,
    }


def test_immutable_release_tag_ruleset_is_exact() -> None:
    payload = load_receipt()
    assert payload["tag_ruleset"] == {"action": "created", "id": 20334249}
    ruleset = payload["post_state"]["immutable_release_tag_ruleset"]["payload"]
    assert ruleset["id"] == 20334249
    assert ruleset["name"] == "Immutable release tags"
    assert ruleset["target"] == "tag"
    assert ruleset["enforcement"] == "active"
    assert ruleset["bypass_actors"] == []
    assert ruleset["conditions"] == {
        "ref_name": {"exclude": [], "include": ["refs/tags/*"]}
    }
    assert ruleset["rules"] == [{"type": "deletion"}, {"type": "non_fast_forward"}]


def test_security_controls_and_codeql_are_exact() -> None:
    post = load_receipt()["post_state"]
    assert post["vulnerability_alerts"]["status"] == 204
    assert post["automated_security_fixes"]["payload"] == {
        "enabled": True,
        "paused": False,
    }
    assert post["private_vulnerability_reporting"]["payload"] == {"enabled": True}
    codeql = post["codeql_default_setup"]["payload"]
    assert codeql["state"] == "configured"
    assert set(codeql["languages"]) == {"actions", "python"}
    assert codeql["query_suite"] == "extended"
    assert codeql["threat_model"] == "remote"
    assert codeql["runner_type"] == "standard"
    assert load_receipt()["mutations"][-1]["payload"] == {
        "run_id": 30859081732,
        "run_url": (
            "https://api.github.com/repos/grandchallenge/MODULUS/actions/runs/30859081732"
        ),
    }


def test_validation_and_claim_boundaries_remain_closed() -> None:
    payload = load_receipt()
    assert payload["validation"] == {
        "codeql_default_setup": True,
        "dependabot_security_updates": True,
        "immutable_release_tags": True,
        "main_ruleset": True,
        "organization_wide_conformance": False,
        "private_vulnerability_reporting": True,
        "repository_merge_settings": True,
        "vulnerability_alerts": True,
    }
    assert payload["claim_boundaries"] == {
        "certification_claims_authorized": False,
        "commercial_claims_authorized": False,
        "deployment_claims_authorized": False,
        "mathematical_claims_authorized": False,
        "novelty_claims_authorized": False,
        "organization_wide_conformance": False,
    }


def test_no_credentials_are_embedded() -> None:
    payload = load_receipt()
    serialized = json.dumps(payload)
    assert not re.search(
        r"(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        serialized,
    )
    for key in ("authorization", "access_token", "token", "gh_token", "github_token"):
        assert f'"{key}":' not in serialized.lower()
