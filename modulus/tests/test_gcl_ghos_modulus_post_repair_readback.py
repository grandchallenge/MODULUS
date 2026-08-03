from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "gcl_ghos_modulus_post_repair_readback.py"
)
SPEC = importlib.util.spec_from_file_location("modulus_post_repair_readback", MODULE_PATH)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def endpoint(path, payload, status=200):
    return {"method": "GET", "path": path, "status": status, "payload": payload}


def surface(path, content):
    raw = content.encode("utf-8")
    return {
        "path": path,
        "blob_sha": "a" * 40,
        "sha256": M.sha256_bytes(raw),
        "size": len(raw),
        "content_utf8": content,
    }


def valid_value():
    main = M.DEFAULT_EXPECTED_MAIN
    main_rule = M.main_ruleset_expected()
    tag_rule = M.tag_ruleset_expected()
    list_payload = [
        {"id": M.AUTHORITY["main_ruleset_id"]},
        {"id": M.AUTHORITY["tag_ruleset_id"]},
    ]
    workflows = {
        path: {"path": path, "state": "active"} for path in M.WORKFLOW_PATHS
    }
    checks = [
        {
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "head_sha": main,
        }
        for name in M.REQUIRED_CONTEXTS
    ]
    ci = "\n".join(
        [
            "uses: actions/checkout@" + "1" * 40,
            "uses: actions/setup-python@" + "2" * 40,
            "uses: actions/upload-artifact@" + "3" * 40,
        ]
    )
    gcl = "\n".join(
        [
            "profile: Provider",
            "standards_ref: " + "4" * 40,
            "uses: grandchallenge/.github/.github/workflows/gcl-policy.yml@" + "5" * 40,
            "uses: grandchallenge/.github/.github/workflows/gcl-security.yml@" + "6" * 40,
        ]
    )
    surfaces = [
        surface(
            ".github/CODEOWNERS",
            "@grandchallenge/provider-maintainers\n"
            "@grandchallenge/security\n"
            "@grandchallenge/the-council\n"
            "@grandchallenge/amanuensis\n",
        ),
        surface(
            ".github/dependabot.yml",
            "package-ecosystem: github-actions\npackage-ecosystem: pip\n",
        ),
        surface(".github/workflows/ci.yml", ci),
        surface(".github/workflows/gcl-conformance.yml", gcl),
        surface("AGENTS.md", "bounded agent authority\n"),
        surface("SUPPORT.md", "governed support\n"),
    ]
    metadata = {
        "full_name": M.REPOSITORY,
        "default_branch": "main",
        "permissions": {"admin": True},
        "allow_auto_merge": True,
        "allow_merge_commit": True,
        "allow_rebase_merge": False,
        "allow_squash_merge": True,
        "allow_update_branch": False,
        "delete_branch_on_merge": True,
    }
    ref_payload = {"object": {"sha": main}}
    return {
        "$schema": M.SCHEMA_PATH,
        "schema_version": M.SCHEMA_VERSION,
        "operation_id": M.OPERATION_ID,
        "recorded_at": "2026-08-03T23:00:00Z",
        "api_version": M.API_VERSION,
        "repository": M.REPOSITORY,
        "authority": copy.deepcopy(M.AUTHORITY),
        "actor": {"login": "fyremael", "id": 17925951, "repository_admin": True},
        "protected_main": {
            "expected_sha": main,
            "start_ref": endpoint("/ref", ref_payload),
            "end_ref": endpoint("/ref", ref_payload),
        },
        "repository_metadata": endpoint("/repo", metadata),
        "repository_rulesets": {
            "list": endpoint(
                f"/repos/{M.REPOSITORY}/rulesets?per_page=100&includes_parents=true",
                list_payload,
            ),
            "details": [
                endpoint(
                    f"/repos/{M.REPOSITORY}/rulesets/{M.AUTHORITY['main_ruleset_id']}",
                    main_rule,
                ),
                endpoint(
                    f"/repos/{M.REPOSITORY}/rulesets/{M.AUTHORITY['tag_ruleset_id']}",
                    tag_rule,
                ),
            ],
        },
        "repository_settings": {
            "actions_permissions": endpoint("/actions", {}),
            "actions_workflow_permissions": endpoint("/workflow", {}),
            "main_protection": endpoint("/protection", None, 404),
        },
        "security_controls": {
            "vulnerability_alerts": endpoint("/alerts", None, 204),
            "automated_security_fixes": endpoint(
                "/fixes", {"enabled": True, "paused": False}
            ),
            "private_vulnerability_reporting": endpoint(
                "/pvr", {"enabled": True}
            ),
            "codeql_default_setup": endpoint(
                "/codeql",
                {
                    "state": "configured",
                    "languages": ["actions", "python"],
                    "query_suite": "extended",
                    "threat_model": "remote",
                    "runner_type": "standard",
                    "schedule": "weekly",
                },
            ),
        },
        "workflow_inventory": {
            "endpoint": endpoint("/workflows", {"workflows": list(workflows.values())}),
            "required_workflows": workflows,
        },
        "protected_main_check_runs": {
            "endpoint": endpoint("/checks", {"check_runs": checks}),
            "required_contexts": checks,
        },
        "surfaces": surfaces,
        "validation": {
            "settings_match_authorized_target": True,
            "rulesets_match_authorized_target": True,
            "security_controls_match_authorized_target": True,
            "required_workflows_active": True,
            "required_checks_successful": True,
            "required_surfaces_present": True,
            "modulus_deviation_disposition_authorized": False,
            "organization_wide_conformance": False,
        },
        "claim_boundaries": copy.deepcopy(M.CLAIM_BOUNDARIES),
    }


def reject(mutator):
    value = valid_value()
    mutator(value)
    with pytest.raises(M.ReadbackError):
        M.validate_readback(value)


def test_valid_fixture():
    M.validate_readback(valid_value())


def test_canonical_bytes_are_stable():
    value = valid_value()
    assert M.canonical_bytes(value) == M.canonical_bytes(json.loads(M.canonical_bytes(value)))


def test_rejects_authority_drift():
    reject(lambda value: value["authority"].update({"main_ruleset_id": 1}))


def test_rejects_protected_main_drift():
    reject(
        lambda value: value["protected_main"]["end_ref"]["payload"]["object"].update(
            {"sha": "0" * 40}
        )
    )


def test_rejects_merge_setting_drift():
    reject(
        lambda value: value["repository_metadata"]["payload"].update(
            {"allow_rebase_merge": True}
        )
    )


def test_rejects_ruleset_identity_mismatch():
    reject(
        lambda value: value["repository_rulesets"]["details"][0]["payload"].update(
            {"id": 9}
        )
    )


def test_rejects_main_ruleset_drift():
    reject(
        lambda value: value["repository_rulesets"]["details"][0]["payload"]["rules"][2][
            "parameters"
        ].update({"required_approving_review_count": 1})
    )


def test_rejects_tag_ruleset_drift():
    reject(
        lambda value: value["repository_rulesets"]["details"][1]["payload"].update(
            {"bypass_actors": [{"actor_id": 1}]}
        )
    )


def test_rejects_security_control_drift():
    reject(
        lambda value: value["security_controls"]["automated_security_fixes"][
            "payload"
        ].update({"enabled": False})
    )


def test_rejects_missing_workflow():
    reject(
        lambda value: value["workflow_inventory"]["required_workflows"].pop(
            ".github/workflows/ci.yml"
        )
    )


def test_rejects_failed_required_context():
    reject(
        lambda value: value["protected_main_check_runs"]["required_contexts"][0].update(
            {"conclusion": "failure"}
        )
    )


def test_rejects_surface_hash_drift():
    reject(lambda value: value["surfaces"][0].update({"sha256": "0" * 64}))


def test_rejects_mutable_action_reference():
    reject(
        lambda value: value["surfaces"][2].update(
            {
                "content_utf8": "uses: actions/checkout@v4\n",
                "sha256": M.sha256_bytes(b"uses: actions/checkout@v4\n"),
                "size": len(b"uses: actions/checkout@v4\n"),
            }
        )
    )


def test_rejects_claim_promotion():
    reject(
        lambda value: value["claim_boundaries"].update(
            {"organization_wide_conformance": True}
        )
    )


def test_rejects_credential_material():
    value = valid_value()
    token = "ghp_" + "A" * 30
    value["surfaces"][4]["content_utf8"] = token
    raw = token.encode()
    value["surfaces"][4]["sha256"] = M.sha256_bytes(raw)
    value["surfaces"][4]["size"] = len(raw)
    with pytest.raises(M.ReadbackError):
        M.validate_readback(value)
