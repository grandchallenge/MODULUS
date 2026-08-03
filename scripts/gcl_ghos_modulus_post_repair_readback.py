#!/usr/bin/env python3
"""Collect or validate GCL-GHOS-MODULUS-POST-REPAIR-READBACK-001.

The collector is GET-only. It binds the exact protected main head, repository
settings, rulesets, security controls, workflow inventory/check runs, and
governed repository surfaces. It emits canonical JSON and a SHA-256 companion.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

API_ROOT = "https://api.github.com"
API_VERSION = "2026-03-10"
OWNER = "grandchallenge"
REPO = "MODULUS"
REPOSITORY = f"{OWNER}/{REPO}"
OPERATION_ID = "GCL-GHOS-MODULUS-POST-REPAIR-READBACK-001"
SCHEMA_PATH = (
    "governance/settings-readback/"
    "GCL-GHOS-MODULUS-POST-REPAIR-READBACK-001.schema.json"
)
SCHEMA_VERSION = "1.1.0"
OWNER_CONTROLS_EVIDENCE_MERGE = "f54dd2c0b26ea46ef6b598f6a65dfcef2c47da47"

AUTHORITY = {
    "parent_issue": "grandchallenge/MODULUS#13",
    "owner_controls_issue": "grandchallenge/MODULUS#10",
    "remediation_issue": "grandchallenge/MODULUS#6",
    "campaign_issue": "grandchallenge/gcl-standards#22",
    "profile_merge": "c027ceafe1a2226ce8abeec36c739aeaa45ec784",
    "surfaces_merge": "7f9efc5a818655454e6117baa596e8382221a874",
    "owner_controls_evidence_merge": OWNER_CONTROLS_EVIDENCE_MERGE,
    "main_ruleset_id": 20266757,
    "tag_ruleset_id": 20334249,
}

SURFACE_PATHS = (
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/gcl-conformance.yml",
    "AGENTS.md",
    "SUPPORT.md",
)

WORKFLOW_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/gcl-conformance.yml",
)

REQUIRED_CONTEXTS = (
    "test-and-lint (3.10)",
    "test-and-lint (3.11)",
    "test-and-lint (3.12)",
    "benchmark-report",
    "policy / policy",
    "security / action-policy",
)

CLAIM_BOUNDARIES = {
    "organization_wide_conformance": False,
    "modulus_deviation_disposition_authorized": False,
    "mathematical_claims_authorized": False,
    "certification_claims_authorized": False,
    "novelty_claims_authorized": False,
    "deployment_claims_authorized": False,
    "commercial_claims_authorized": False,
}

RequestFn = Callable[[str, set[int]], dict[str, Any]]


class ReadbackError(ValueError):
    """Raised when a readback is incomplete, ambiguous, or inconsistent."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_token() -> str:
    token = os.environ.get("GH_TOKEN", "").strip()
    if token:
        return token
    proc = subprocess.run(
        ["gh", "auth", "token"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ReadbackError(
            "Unable to obtain an active GitHub CLI credential. "
            "Authenticate with `gh auth login` or set GH_TOKEN."
        )
    token = proc.stdout.strip()
    if not token:
        raise ReadbackError("GitHub CLI returned an empty credential.")
    return token


def make_requester(token: str) -> RequestFn:
    if not token:
        raise ReadbackError("A GitHub credential is required.")

    def request(path: str, allowed: set[int] = {200}) -> dict[str, Any]:
        req = urllib.request.Request(
            API_ROOT + path,
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": OPERATION_ID,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
        except urllib.error.URLError as exc:
            raise ReadbackError(f"GET {path} failed before an HTTP response: {exc}") from exc

        payload: Any = None
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {"raw_text": raw.decode("utf-8", errors="replace")}

        if status not in allowed:
            raise ReadbackError(
                f"GET {path} returned HTTP {status}; allowed={sorted(allowed)}; "
                f"payload={payload!r}"
            )
        return {"method": "GET", "path": path, "status": status, "payload": payload}

    return request


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReadbackError(f"{name} must be an object")
    return value


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReadbackError(f"{name} must be an array")
    return value


def main_ruleset_expected() -> dict[str, Any]:
    return {
        "id": AUTHORITY["main_ruleset_id"],
        "name": "GCL protected main",
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "allowed_merge_methods": ["merge", "squash"],
                    "dismiss_stale_reviews_on_push": True,
                    "dismissal_restriction": {
                        "allowed_actors": [],
                        "enabled": False,
                    },
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                    "required_approving_review_count": 0,
                    "required_review_thread_resolution": True,
                    "required_reviewers": [],
                },
            },
            {
                "type": "required_status_checks",
                "parameters": {
                    "do_not_enforce_on_create": False,
                    "strict_required_status_checks_policy": True,
                    "required_status_checks": [
                        {"context": context} for context in REQUIRED_CONTEXTS
                    ],
                },
            },
        ],
    }


def tag_ruleset_expected() -> dict[str, Any]:
    return {
        "id": AUTHORITY["tag_ruleset_id"],
        "name": "Immutable release tags",
        "target": "tag",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {"ref_name": {"include": ["refs/tags/*"], "exclude": []}},
        "rules": [{"type": "deletion"}, {"type": "non_fast_forward"}],
    }


def normalized_rules(value: Any) -> list[dict[str, Any]]:
    rules = require_list(value, "rules")
    copied = json.loads(json.dumps(rules))
    for rule in copied:
        if not isinstance(rule, dict):
            raise ReadbackError("rules contain a non-object")
        parameters = rule.get("parameters")
        if isinstance(parameters, dict):
            checks = parameters.get("required_status_checks")
            if isinstance(checks, list):
                for check in checks:
                    if isinstance(check, dict):
                        check.pop("integration_id", None)
    return copied


def validate_ruleset(actual: Any, expected: dict[str, Any], name: str) -> None:
    actual = require_object(actual, name)
    for key in ("id", "name", "target", "enforcement", "conditions"):
        if actual.get(key) != expected[key]:
            raise ReadbackError(f"{name} {key} mismatch")
    if actual.get("bypass_actors", []) != []:
        raise ReadbackError(f"{name} contains bypass actors")
    if normalized_rules(actual.get("rules")) != normalized_rules(expected["rules"]):
        raise ReadbackError(f"{name} rules mismatch")


def collect_rulesets(request: RequestFn) -> dict[str, Any]:
    prefix = f"/repos/{REPOSITORY}/rulesets"
    listed = request(f"{prefix}?per_page=100&includes_parents=true", {200})
    items = require_list(listed["payload"], "ruleset list")
    details: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in items:
        item = require_object(item, "ruleset list item")
        rule_id = item.get("id")
        if not isinstance(rule_id, int):
            raise ReadbackError("ruleset list item has no integer id")
        if rule_id in seen:
            raise ReadbackError(f"duplicate ruleset id {rule_id}")
        seen.add(rule_id)
        details.append(request(f"{prefix}/{rule_id}", {200}))
    return {"list": listed, "details": details}


def validate_ruleset_collection(value: Any) -> dict[int, dict[str, Any]]:
    value = require_object(value, "rulesets")
    listed = require_object(value.get("list"), "ruleset list endpoint")
    if listed.get("status") != 200:
        raise ReadbackError("ruleset list status mismatch")
    items = require_list(listed.get("payload"), "ruleset list payload")
    listed_ids: list[int] = []
    for item in items:
        item = require_object(item, "ruleset list item")
        rule_id = item.get("id")
        if not isinstance(rule_id, int):
            raise ReadbackError("ruleset list item id is invalid")
        listed_ids.append(rule_id)
    if len(listed_ids) != len(set(listed_ids)):
        raise ReadbackError("ruleset list contains duplicate ids")

    details = require_list(value.get("details"), "ruleset details")
    detail_map: dict[int, dict[str, Any]] = {}
    for endpoint in details:
        endpoint = require_object(endpoint, "ruleset detail endpoint")
        if endpoint.get("status") != 200:
            raise ReadbackError("ruleset detail status mismatch")
        payload = require_object(endpoint.get("payload"), "ruleset detail payload")
        rule_id = payload.get("id")
        if not isinstance(rule_id, int):
            raise ReadbackError("ruleset detail id is invalid")
        if rule_id in detail_map:
            raise ReadbackError("ruleset details contain duplicate ids")
        if endpoint.get("path") != f"/repos/{REPOSITORY}/rulesets/{rule_id}":
            raise ReadbackError("ruleset detail path mismatch")
        detail_map[rule_id] = payload
    if set(listed_ids) != set(detail_map):
        raise ReadbackError("ruleset list/detail identities do not match")
    if set(detail_map) != {
        AUTHORITY["main_ruleset_id"],
        AUTHORITY["tag_ruleset_id"],
    }:
        raise ReadbackError("unexpected or missing repository ruleset identity")
    return detail_map


def decode_content_endpoint(endpoint: dict[str, Any], expected_path: str) -> dict[str, Any]:
    if endpoint.get("status") != 200:
        raise ReadbackError(f"{expected_path} content endpoint failed")
    payload = require_object(endpoint.get("payload"), f"{expected_path} payload")
    if payload.get("path") != expected_path or payload.get("type") != "file":
        raise ReadbackError(f"{expected_path} content identity mismatch")
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise ReadbackError(f"{expected_path} content encoding mismatch")
    try:
        raw = base64.b64decode(payload["content"], validate=False)
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReadbackError(f"{expected_path} is not valid UTF-8 content") from exc
    blob = payload.get("sha")
    if not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise ReadbackError(f"{expected_path} blob identity is invalid")
    return {
        "path": expected_path,
        "blob_sha": blob,
        "sha256": sha256_bytes(raw),
        "size": len(raw),
        "content_utf8": text,
    }


def collect_surfaces(request: RequestFn, ref: str) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for path in SURFACE_PATHS:
        quoted = urllib.parse.quote(path, safe="/")
        endpoint = request(
            f"/repos/{REPOSITORY}/contents/{quoted}?ref={urllib.parse.quote(ref)}",
            {200},
        )
        surfaces.append(decode_content_endpoint(endpoint, path))
    return surfaces


def validate_surface_semantics(surface_map: dict[str, dict[str, Any]]) -> None:
    codeowners = surface_map[".github/CODEOWNERS"]["content_utf8"]
    for required in (
        "@grandchallenge/provider-maintainers",
        "@grandchallenge/security",
        "@grandchallenge/the-council",
        "@grandchallenge/amanuensis",
    ):
        if required not in codeowners:
            raise ReadbackError(f"CODEOWNERS is missing {required}")

    dependabot = surface_map[".github/dependabot.yml"]["content_utf8"]
    if "package-ecosystem: github-actions" not in dependabot:
        raise ReadbackError("Dependabot lacks GitHub Actions coverage")
    if "package-ecosystem: pip" not in dependabot:
        raise ReadbackError("Dependabot lacks pip coverage")

    for path in ("AGENTS.md", "SUPPORT.md"):
        if not surface_map[path]["content_utf8"].strip():
            raise ReadbackError(f"{path} is empty")

    for path in WORKFLOW_PATHS:
        workflow = surface_map[path]["content_utf8"]
        for match in re.finditer(r"^\s*uses:\s*([^\s#]+)", workflow, re.MULTILINE):
            reference = match.group(1)
            if reference.startswith("./"):
                continue
            if "@" not in reference:
                raise ReadbackError(f"{path} contains malformed action reference")
            suffix = reference.rsplit("@", 1)[1]
            if not re.fullmatch(r"[0-9a-f]{40}", suffix):
                raise ReadbackError(f"{path} contains mutable action reference {reference}")

    gcl = surface_map[".github/workflows/gcl-conformance.yml"]["content_utf8"]
    if "profile: Provider" not in gcl:
        raise ReadbackError("GCL workflow does not bind Provider profile")
    match = re.search(r"standards_ref:\s*([0-9a-f]{40})", gcl)
    if not match:
        raise ReadbackError("GCL workflow lacks a pinned standards_ref")


def collect_workflows(request: RequestFn) -> dict[str, Any]:
    endpoint = request(
        f"/repos/{REPOSITORY}/actions/workflows?per_page=100",
        {200},
    )
    payload = require_object(endpoint["payload"], "workflow inventory payload")
    workflows = require_list(payload.get("workflows"), "workflow inventory")
    selected: dict[str, dict[str, Any]] = {}
    for workflow in workflows:
        workflow = require_object(workflow, "workflow inventory item")
        path = workflow.get("path")
        if path in WORKFLOW_PATHS:
            if path in selected:
                raise ReadbackError(f"duplicate workflow inventory path {path}")
            selected[path] = workflow
    if set(selected) != set(WORKFLOW_PATHS):
        raise ReadbackError("required workflow inventory is incomplete")
    for path, workflow in selected.items():
        if workflow.get("state") != "active":
            raise ReadbackError(f"{path} workflow is not active")
    return {"endpoint": endpoint, "required_workflows": selected}


def collect_check_runs(
    request: RequestFn,
    commit_sha: str,
    *,
    attempts: int,
    interval_seconds: float,
) -> dict[str, Any]:
    path = f"/repos/{REPOSITORY}/commits/{commit_sha}/check-runs?per_page=100"
    last: dict[str, Any] | None = None
    for index in range(attempts):
        last = request(path, {200})
        payload = require_object(last["payload"], "check-runs payload")
        check_runs = require_list(payload.get("check_runs"), "check-runs")
        successful = {
            row.get("name")
            for row in check_runs
            if isinstance(row, dict)
            and row.get("status") == "completed"
            and row.get("conclusion") == "success"
        }
        if set(REQUIRED_CONTEXTS).issubset(successful):
            selected = [
                row
                for row in check_runs
                if isinstance(row, dict) and row.get("name") in REQUIRED_CONTEXTS
            ]
            return {"endpoint": last, "required_contexts": selected}
        if index + 1 < attempts:
            time.sleep(interval_seconds)
    raise ReadbackError(
        "required protected-main check contexts did not all reach success; "
        f"last={last!r}"
    )


def collect_readback(
    request: RequestFn,
    *,
    expected_main: str,
    check_attempts: int,
    check_interval_seconds: float,
) -> dict[str, Any]:
    actor_endpoint = request("/user", {200})
    actor = require_object(actor_endpoint["payload"], "actor")
    login = actor.get("login")
    if not isinstance(login, str) or not login:
        raise ReadbackError("actor login is missing")

    metadata = request(f"/repos/{REPOSITORY}", {200})
    repository = require_object(metadata["payload"], "repository metadata")
    permissions = repository.get("permissions")
    if not isinstance(permissions, dict) or permissions.get("admin") is not True:
        raise ReadbackError("repository-admin proof is absent")
    if repository.get("full_name") != REPOSITORY:
        raise ReadbackError("repository identity mismatch")
    if repository.get("default_branch") != "main":
        raise ReadbackError("default branch is not main")

    start_ref = request(f"/repos/{REPOSITORY}/git/ref/heads/main", {200})
    start_ref_payload = require_object(start_ref["payload"], "main ref")
    start_object = require_object(start_ref_payload.get("object"), "main ref object")
    main_sha = start_object.get("sha")
    if main_sha != expected_main:
        raise ReadbackError(
            f"protected main moved: expected {expected_main}, observed {main_sha}"
        )

    baseline_compare = request(
        f"/repos/{REPOSITORY}/compare/"
        f"{OWNER_CONTROLS_EVIDENCE_MERGE}...{expected_main}",
        {200},
    )

    surfaces = collect_surfaces(request, main_sha)
    workflows = collect_workflows(request)
    check_runs = collect_check_runs(
        request,
        main_sha,
        attempts=check_attempts,
        interval_seconds=check_interval_seconds,
    )

    result = {
        "$schema": SCHEMA_PATH,
        "schema_version": SCHEMA_VERSION,
        "operation_id": OPERATION_ID,
        "recorded_at": utc_now(),
        "api_version": API_VERSION,
        "repository": REPOSITORY,
        "authority": dict(AUTHORITY),
        "actor": {
            "login": login,
            "id": actor.get("id"),
            "repository_admin": True,
        },
        "protected_main": {
            "expected_sha": expected_main,
            "authority_baseline_compare": baseline_compare,
            "start_ref": start_ref,
        },
        "repository_metadata": metadata,
        "repository_rulesets": collect_rulesets(request),
        "repository_settings": {
            "actions_permissions": request(
                f"/repos/{REPOSITORY}/actions/permissions", {200}
            ),
            "actions_workflow_permissions": request(
                f"/repos/{REPOSITORY}/actions/permissions/workflow", {200}
            ),
            "main_protection": request(
                f"/repos/{REPOSITORY}/branches/main/protection", {200, 404}
            ),
        },
        "security_controls": {
            "vulnerability_alerts": request(
                f"/repos/{REPOSITORY}/vulnerability-alerts", {204}
            ),
            "automated_security_fixes": request(
                f"/repos/{REPOSITORY}/automated-security-fixes", {200}
            ),
            "private_vulnerability_reporting": request(
                f"/repos/{REPOSITORY}/private-vulnerability-reporting", {200}
            ),
            "codeql_default_setup": request(
                f"/repos/{REPOSITORY}/code-scanning/default-setup", {200}
            ),
        },
        "workflow_inventory": workflows,
        "protected_main_check_runs": check_runs,
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
        "claim_boundaries": dict(CLAIM_BOUNDARIES),
    }

    end_ref = request(f"/repos/{REPOSITORY}/git/ref/heads/main", {200})
    end_payload = require_object(end_ref["payload"], "final main ref")
    end_object = require_object(end_payload.get("object"), "final main ref object")
    if end_object.get("sha") != main_sha:
        raise ReadbackError("protected main moved during collection")
    result["protected_main"]["end_ref"] = end_ref

    validate_readback(result)
    return result


def endpoint_payload(endpoint: Any, name: str, status: int) -> Any:
    endpoint = require_object(endpoint, name)
    if endpoint.get("method") != "GET" or endpoint.get("status") != status:
        raise ReadbackError(f"{name} endpoint mismatch")
    if not isinstance(endpoint.get("path"), str):
        raise ReadbackError(f"{name} path is missing")
    if "payload" not in endpoint:
        raise ReadbackError(f"{name} payload is missing")
    return endpoint["payload"]


def validate_repository_settings(metadata: dict[str, Any]) -> None:
    expected = {
        "allow_auto_merge": True,
        "allow_merge_commit": True,
        "allow_rebase_merge": False,
        "allow_squash_merge": True,
        "allow_update_branch": False,
        "delete_branch_on_merge": True,
    }
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise ReadbackError(
            f"repository merge settings mismatch: expected={expected}, actual={observed}"
        )


def validate_readback(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ReadbackError("readback must be an object")
    if value.get("$schema") != SCHEMA_PATH:
        raise ReadbackError("schema identity mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ReadbackError("schema version mismatch")
    if value.get("operation_id") != OPERATION_ID:
        raise ReadbackError("operation identity mismatch")
    if value.get("api_version") != API_VERSION:
        raise ReadbackError("API version mismatch")
    if value.get("repository") != REPOSITORY:
        raise ReadbackError("repository identity mismatch")
    if value.get("authority") != AUTHORITY:
        raise ReadbackError("authority identity mismatch")
    if value.get("claim_boundaries") != CLAIM_BOUNDARIES:
        raise ReadbackError("claim boundaries must remain closed")

    actor = require_object(value.get("actor"), "actor")
    if actor.get("repository_admin") is not True:
        raise ReadbackError("repository-admin proof is absent")
    if not isinstance(actor.get("login"), str) or not actor["login"]:
        raise ReadbackError("actor login is missing")

    protected_main = require_object(value.get("protected_main"), "protected main")
    expected_main = protected_main.get("expected_sha")
    if not isinstance(expected_main, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_main):
        raise ReadbackError("protected main expected identity is invalid")
    for key in ("start_ref", "end_ref"):
        payload = endpoint_payload(protected_main.get(key), key, 200)
        payload = require_object(payload, key)
        obj = require_object(payload.get("object"), f"{key} object")
        if obj.get("sha") != expected_main:
            raise ReadbackError(f"{key} protected main identity drift")

    compare = endpoint_payload(
        protected_main.get("authority_baseline_compare"),
        "authority baseline compare",
        200,
    )
    compare = require_object(compare, "authority baseline compare payload")
    expected_compare_path = (
        f"/repos/{REPOSITORY}/compare/"
        f"{OWNER_CONTROLS_EVIDENCE_MERGE}...{expected_main}"
    )
    compare_endpoint = require_object(
        protected_main.get("authority_baseline_compare"),
        "authority baseline compare endpoint",
    )
    if compare_endpoint.get("path") != expected_compare_path:
        raise ReadbackError("authority baseline compare path mismatch")
    if compare.get("status") not in {"ahead", "identical"}:
        raise ReadbackError("protected main does not descend from admitted evidence merge")
    if compare.get("behind_by") != 0:
        raise ReadbackError("protected main is behind the admitted evidence merge")
    ahead_by = compare.get("ahead_by")
    if not isinstance(ahead_by, int) or ahead_by < 0:
        raise ReadbackError("authority baseline compare ahead count is invalid")
    base_commit = require_object(compare.get("base_commit"), "compare base commit")
    merge_base = require_object(compare.get("merge_base_commit"), "compare merge base")
    if base_commit.get("sha") != OWNER_CONTROLS_EVIDENCE_MERGE:
        raise ReadbackError("authority baseline compare base identity drift")
    if merge_base.get("sha") != OWNER_CONTROLS_EVIDENCE_MERGE:
        raise ReadbackError("authority baseline compare merge-base drift")
    if compare.get("status") == "identical" and expected_main != OWNER_CONTROLS_EVIDENCE_MERGE:
        raise ReadbackError("identical compare does not bind the expected main")
    if compare.get("status") == "ahead" and expected_main == OWNER_CONTROLS_EVIDENCE_MERGE:
        raise ReadbackError("ahead compare cannot bind the baseline itself")

    metadata = endpoint_payload(
        value.get("repository_metadata"), "repository metadata", 200
    )
    metadata = require_object(metadata, "repository metadata payload")
    if metadata.get("full_name") != REPOSITORY or metadata.get("default_branch") != "main":
        raise ReadbackError("repository metadata identity drift")
    permissions = metadata.get("permissions")
    if not isinstance(permissions, dict) or permissions.get("admin") is not True:
        raise ReadbackError("metadata repository-admin proof is absent")
    validate_repository_settings(metadata)

    details = validate_ruleset_collection(value.get("repository_rulesets"))
    validate_ruleset(
        details[AUTHORITY["main_ruleset_id"]],
        main_ruleset_expected(),
        "main ruleset",
    )
    validate_ruleset(
        details[AUTHORITY["tag_ruleset_id"]],
        tag_ruleset_expected(),
        "tag ruleset",
    )

    settings = require_object(value.get("repository_settings"), "repository settings")
    endpoint_payload(settings.get("actions_permissions"), "Actions permissions", 200)
    endpoint_payload(
        settings.get("actions_workflow_permissions"),
        "workflow permissions",
        200,
    )
    main_protection = require_object(
        settings.get("main_protection"), "main protection endpoint"
    )
    if main_protection.get("status") not in (200, 404):
        raise ReadbackError("main protection status is unsupported")

    security = require_object(value.get("security_controls"), "security controls")
    endpoint_payload(
        security.get("vulnerability_alerts"), "vulnerability alerts", 204
    )
    fixes = endpoint_payload(
        security.get("automated_security_fixes"),
        "automated security fixes",
        200,
    )
    fixes = require_object(fixes, "automated security fixes payload")
    if fixes.get("enabled") is not True or fixes.get("paused") is True:
        raise ReadbackError("Dependabot security updates are not enabled")
    pvr = endpoint_payload(
        security.get("private_vulnerability_reporting"),
        "private vulnerability reporting",
        200,
    )
    pvr = require_object(pvr, "private vulnerability reporting payload")
    if pvr.get("enabled") is not True:
        raise ReadbackError("private vulnerability reporting is not enabled")
    codeql = endpoint_payload(
        security.get("codeql_default_setup"), "CodeQL default setup", 200
    )
    codeql = require_object(codeql, "CodeQL payload")
    if (
        codeql.get("state") != "configured"
        or set(codeql.get("languages", [])) != {"actions", "python"}
        or codeql.get("query_suite") != "extended"
        or codeql.get("threat_model") != "remote"
        or codeql.get("runner_type") != "standard"
        or codeql.get("schedule") != "weekly"
    ):
        raise ReadbackError("CodeQL default setup drift")

    workflow_inventory = require_object(
        value.get("workflow_inventory"), "workflow inventory"
    )
    endpoint_payload(
        workflow_inventory.get("endpoint"), "workflow inventory endpoint", 200
    )
    required_workflows = require_object(
        workflow_inventory.get("required_workflows"), "required workflows"
    )
    if set(required_workflows) != set(WORKFLOW_PATHS):
        raise ReadbackError("required workflow inventory drift")
    for path, workflow in required_workflows.items():
        workflow = require_object(workflow, path)
        if workflow.get("path") != path or workflow.get("state") != "active":
            raise ReadbackError(f"{path} workflow state drift")

    checks = require_object(
        value.get("protected_main_check_runs"), "protected main check runs"
    )
    endpoint_payload(checks.get("endpoint"), "check-runs endpoint", 200)
    rows = require_list(checks.get("required_contexts"), "required contexts")
    successful = {
        row.get("name")
        for row in rows
        if isinstance(row, dict)
        and row.get("status") == "completed"
        and row.get("conclusion") == "success"
        and row.get("head_sha") == expected_main
    }
    if successful != set(REQUIRED_CONTEXTS):
        raise ReadbackError("required check contexts are incomplete or unsuccessful")

    surfaces = require_list(value.get("surfaces"), "surfaces")
    if len(surfaces) != len(SURFACE_PATHS):
        raise ReadbackError("surface count mismatch")
    surface_map: dict[str, dict[str, Any]] = {}
    for surface in surfaces:
        surface = require_object(surface, "surface")
        path = surface.get("path")
        if path in surface_map:
            raise ReadbackError("duplicate surface path")
        if path not in SURFACE_PATHS:
            raise ReadbackError("unexpected surface path")
        content = surface.get("content_utf8")
        if not isinstance(content, str):
            raise ReadbackError(f"{path} content is missing")
        raw = content.encode("utf-8")
        if surface.get("size") != len(raw):
            raise ReadbackError(f"{path} size mismatch")
        if surface.get("sha256") != sha256_bytes(raw):
            raise ReadbackError(f"{path} SHA-256 mismatch")
        if not isinstance(surface.get("blob_sha"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", surface["blob_sha"]
        ):
            raise ReadbackError(f"{path} blob identity is invalid")
        surface_map[path] = surface
    if set(surface_map) != set(SURFACE_PATHS):
        raise ReadbackError("surface inventory mismatch")
    validate_surface_semantics(surface_map)

    expected_validation = {
        "settings_match_authorized_target": True,
        "rulesets_match_authorized_target": True,
        "security_controls_match_authorized_target": True,
        "required_workflows_active": True,
        "required_checks_successful": True,
        "required_surfaces_present": True,
        "modulus_deviation_disposition_authorized": False,
        "organization_wide_conformance": False,
    }
    if value.get("validation") != expected_validation:
        raise ReadbackError("validation disposition drift")

    serialized = json.dumps(value, sort_keys=True)
    forbidden = (
        r"gh[pousr]_[A-Za-z0-9_]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r'"authorization"\s*:',
    )
    for pattern in forbidden:
        if re.search(pattern, serialized, flags=re.IGNORECASE):
            raise ReadbackError("credential-like material is embedded")


def write_output(value: dict[str, Any], output: Path) -> str:
    data = canonical_bytes(value)
    output.write_bytes(data)
    digest = sha256_bytes(data)
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return digest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"{OPERATION_ID}.json"),
    )
    parser.add_argument("--validate", type=Path)
    parser.add_argument(
        "--expected-main",
        help="exact protected main SHA to bind (required when collecting)",
    )
    parser.add_argument("--check-attempts", type=int, default=30)
    parser.add_argument("--check-interval-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        if args.validate:
            value = json.loads(args.validate.read_text(encoding="utf-8"))
            validate_readback(value)
            raw = args.validate.read_bytes()
            if raw != canonical_bytes(value):
                raise ReadbackError("readback bytes are not canonical")
            digest_path = args.validate.with_suffix(args.validate.suffix + ".sha256")
            expected_text = (
                f"{sha256_bytes(raw)}  {args.validate.name}\n"
            )
            if digest_path.read_text(encoding="utf-8") != expected_text:
                raise ReadbackError("companion digest mismatch")
            print(
                json.dumps(
                    {
                        "validated": str(args.validate),
                        "sha256": sha256_bytes(raw),
                    },
                    sort_keys=True,
                )
            )
            return 0

        if not isinstance(args.expected_main, str) or not re.fullmatch(
            r"[0-9a-f]{40}", args.expected_main
        ):
            raise ReadbackError(
                "--expected-main is required and must be a 40-character lowercase SHA"
            )
        if args.check_attempts < 1:
            raise ReadbackError("--check-attempts must be positive")
        if args.check_interval_seconds < 0:
            raise ReadbackError("--check-interval-seconds cannot be negative")

        request = make_requester(get_token())
        value = collect_readback(
            request,
            expected_main=args.expected_main,
            check_attempts=args.check_attempts,
            check_interval_seconds=args.check_interval_seconds,
        )
        digest = write_output(value, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": digest,
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        ReadbackError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
