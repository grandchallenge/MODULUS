from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = ROOT / "scripts" / "gcl_ghos_modulus_post_repair_readback.py"
EVIDENCE_PATH = (
    ROOT
    / "governance"
    / "settings-readback"
    / "evidence"
    / "GCL-GHOS-MODULUS-POST-REPAIR-READBACK-001.json"
)
DIGEST_PATH = EVIDENCE_PATH.with_suffix(EVIDENCE_PATH.suffix + ".sha256")
EXPECTED_SHA256 = "be189dda6d5ee0b0b9a2d0c9af64f6910215bc24e3e524c9344ef06cbafd9143"
EXPECTED_SIZE = 272012
EXPECTED_MAIN = "d23c285e7f9245c1504a64d108373dccedaf05e6"
BASELINE = "f54dd2c0b26ea46ef6b598f6a65dfcef2c47da47"

SPEC = importlib.util.spec_from_file_location("modulus_post_repair_readback", COLLECTOR_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def load_raw() -> bytes:
    return EVIDENCE_PATH.read_bytes()


def load_value() -> dict:
    return json.loads(load_raw())


def rejected(mutator) -> None:
    value = copy.deepcopy(load_value())
    mutator(value)
    with pytest.raises(M.ReadbackError):
        M.validate_readback(value)


def test_exact_canonical_evidence_identity() -> None:
    raw = load_raw()
    value = json.loads(raw)
    assert len(raw) == EXPECTED_SIZE
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert raw == M.canonical_bytes(value)
    assert DIGEST_PATH.read_text(encoding="utf-8") == (
        f"{EXPECTED_SHA256}  {EVIDENCE_PATH.name}\n"
    )
    M.validate_readback(value)


def test_exact_authority_and_protected_main_binding() -> None:
    value = load_value()
    assert value["schema_version"] == "1.1.0"
    assert value["operation_id"] == "GCL-GHOS-MODULUS-POST-REPAIR-READBACK-001"
    assert value["recorded_at"] == "2026-08-04T00:13:03Z"
    assert value["actor"] == {
        "id": 17925951,
        "login": "fyremael",
        "repository_admin": True,
    }
    protected = value["protected_main"]
    assert protected["expected_sha"] == EXPECTED_MAIN
    assert protected["start_ref"]["payload"]["object"]["sha"] == EXPECTED_MAIN
    assert protected["end_ref"]["payload"]["object"]["sha"] == EXPECTED_MAIN
    compare = protected["authority_baseline_compare"]["payload"]
    assert compare["status"] == "ahead"
    assert compare["ahead_by"] == 20
    assert compare["behind_by"] == 0
    assert compare["base_commit"]["sha"] == BASELINE
    assert compare["merge_base_commit"]["sha"] == BASELINE


def test_exact_ruleset_workflow_check_and_surface_inventories() -> None:
    value = load_value()
    listed = value["repository_rulesets"]["list"]["payload"]
    details = value["repository_rulesets"]["details"]
    assert {row["id"] for row in listed} == {20266757, 20334249}
    assert {row["payload"]["id"] for row in details} == {20266757, 20334249}
    assert set(value["workflow_inventory"]["required_workflows"]) == {
        ".github/workflows/ci.yml",
        ".github/workflows/gcl-conformance.yml",
    }
    contexts = value["protected_main_check_runs"]["required_contexts"]
    assert {row["name"] for row in contexts} == set(M.REQUIRED_CONTEXTS)
    assert all(
        row["status"] == "completed"
        and row["conclusion"] == "success"
        and row["head_sha"] == EXPECTED_MAIN
        for row in contexts
    )
    assert {row["path"] for row in value["surfaces"]} == set(M.SURFACE_PATHS)


def test_security_and_boundaries_remain_closed() -> None:
    value = load_value()
    security = value["security_controls"]
    assert security["vulnerability_alerts"]["status"] == 204
    assert security["automated_security_fixes"]["payload"] == {
        "enabled": True,
        "paused": False,
    }
    assert security["private_vulnerability_reporting"]["payload"]["enabled"] is True
    codeql = security["codeql_default_setup"]["payload"]
    assert codeql["state"] == "configured"
    assert set(codeql["languages"]) == {"actions", "python"}
    assert codeql["query_suite"] == "extended"
    assert codeql["threat_model"] == "remote"
    assert codeql["runner_type"] == "standard"
    assert codeql["schedule"] == "weekly"
    assert all(flag is False for flag in value["claim_boundaries"].values())
    assert value["validation"]["modulus_deviation_disposition_authorized"] is False
    assert value["validation"]["organization_wide_conformance"] is False


def test_rejects_protected_main_drift() -> None:
    rejected(lambda value: value["protected_main"].update({"expected_sha": "0" * 40}))


def test_rejects_ruleset_identity_drift() -> None:
    rejected(
        lambda value: value["repository_rulesets"]["details"][0]["payload"].update(
            {"id": 1}
        )
    )


def test_rejects_failed_required_context() -> None:
    rejected(
        lambda value: value["protected_main_check_runs"]["required_contexts"][0].update(
            {"conclusion": "failure"}
        )
    )


def test_rejects_surface_digest_drift() -> None:
    rejected(lambda value: value["surfaces"][0].update({"sha256": "0" * 64}))


def test_rejects_security_control_drift() -> None:
    rejected(
        lambda value: value["security_controls"]["automated_security_fixes"][
            "payload"
        ].update({"enabled": False})
    )


def test_rejects_claim_promotion() -> None:
    rejected(
        lambda value: value["claim_boundaries"].update(
            {"organization_wide_conformance": True}
        )
    )


def test_evidence_contains_no_credential_material() -> None:
    serialized = json.dumps(load_value(), sort_keys=True)
    forbidden = (
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r'"authorization"\s*:',
    )
    assert not any(re.search(pattern, serialized, re.IGNORECASE) for pattern in forbidden)
