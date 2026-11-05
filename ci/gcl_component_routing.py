#!/usr/bin/env python3
"""GCL-CIR-001 portable, fail-closed component ownership and CI routing guard.

This controller does not alter GitHub branch protection or replace any
repository's native CI. Native workflows retain authority over what executes.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
from pathlib import Path

CONTRACT = "GCL-CIR-001"
VERSION = "1.0.0"
_COMPONENT_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_CONTROL_PREFIXES = (".github/", "ci/gcl_component_routing.py", "governance/ci_components/")


class RoutingError(ValueError):
    pass


def _read(path: Path) -> dict:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RoutingError(f"cannot load {path}: {exc}") from exc
    if not isinstance(row, dict):
        raise RoutingError(f"expected object: {path}")
    return row


def _relative(path: str) -> str:
    if not isinstance(path, str) or not path or path.startswith(("/", "./")):
        raise RoutingError(f"invalid path: {path!r}")
    if "\\" in path or any(part in {"", ".", ".."} for part in path.rstrip("/").split("/")):
        raise RoutingError(f"unsafe path: {path}")
    if any(c in path for c in "*?[]"):
        raise RoutingError(f"glob not permitted in component path: {path}")
    return path


def _components(root: Path) -> dict[str, dict]:
    directory = root / "governance/ci_components"
    files = sorted(directory.glob("*.json"))
    if not files:
        raise RoutingError("missing component manifests")
    components = {}
    for file in files:
        row = _read(file)
        required = {"contract_id", "version", "component_id", "owner", "paths", "depends_on", "checks"}
        if set(row) != required:
            raise RoutingError(f"component keys drift: {file}")
        name = row["component_id"]
        if (not isinstance(name, str) or not _COMPONENT_ID.fullmatch(name)
                or file.stem != name or name in components):
            raise RoutingError(f"component identity drift: {file}")
        if row["contract_id"] != CONTRACT or row["version"] != VERSION:
            raise RoutingError(f"component contract drift: {name}")
        if not isinstance(row["owner"], str) or not row["owner"].strip():
            raise RoutingError(f"missing owner: {name}")
        paths = row["paths"]
        if not isinstance(paths, dict) or set(paths) != {"exact", "prefixes", "test_prefixes"}:
            raise RoutingError(f"invalid path declarations: {name}")
        if not all(isinstance(v, list) and len(v) == len(set(v)) for v in paths.values()):
            raise RoutingError(f"duplicate or invalid path declarations: {name}")
        if not any(paths.values()):
            raise RoutingError(f"component without paths: {name}")
        for kind, values in paths.items():
            for value in values:
                _relative(value)
                if (value.startswith((".github/", "governance/ci_components/"))
                    or (value.startswith("ci/") and (kind == "prefixes" or (kind == "exact" and not value.startswith("ci/test_"))))):
                    raise RoutingError(f"component captured protected control path: {name}: {value}")
                if kind == "prefixes" and not value.endswith("/"):
                    raise RoutingError(f"directory prefix lacks slash: {name}: {value}")
                if kind == "test_prefixes" and (
                    not value.startswith(("tests/test_", "ci/test_", "modulus/tests/test_")) or not value.endswith("_")
                ):
                    raise RoutingError(f"unsafe test prefix: {name}: {value}")
        for key in ("depends_on", "checks"):
            arr = row[key]
            if not isinstance(arr, list) or len(arr) != len(set(arr)) or any(
                not isinstance(v, str) or not v for v in arr
            ):
                raise RoutingError(f"invalid {key}: {name}")
        if not row["checks"]:
            raise RoutingError(f"component has no executing checks: {name}")
        components[name] = row
    for name, row in components.items():
        if name in row["depends_on"]:
            raise RoutingError(f"self dependency: {name}")
        for dep in row["depends_on"]:
            if dep not in components:
                raise RoutingError(f"unknown dependency: {name}->{dep}")
    active, visited = set(), set()

    def visit(name: str) -> None:
        if name in active:
            raise RoutingError(f"dependency cycle: {name}")
        if name in visited:
            return
        active.add(name)
        for upstream in components[name]["depends_on"]:
            visit(upstream)
        active.remove(name)
        visited.add(name)

    for name in sorted(components):
        visit(name)
    return components


def _owner(path: str, components: dict[str, dict]) -> str | None:
    owners = []
    for name, row in components.items():
        decl = row["paths"]
        if (path in decl["exact"]
            or any(path.startswith(p) for p in decl["prefixes"])
            or (path.endswith(".py") and any(path.startswith(p) for p in decl["test_prefixes"]))):
            owners.append(name)
    if len(owners) > 1:
        raise RoutingError(f"ambiguous ownership for {path}: {sorted(owners)}")
    return owners[0] if owners else None


def _check_catalog(root: Path) -> tuple[dict, dict[str, dict]]:
    config = _read(root / "governance/ci_routing_baseline.json")
    if set(config) != {"contract_id", "version", "repository", "mode", "checks", "legacy_test_paths"}:
        raise RoutingError("routing baseline keys drift")
    if config["contract_id"] != CONTRACT or config["version"] != VERSION:
        raise RoutingError("routing contract identity mismatch")
    if config["mode"] != "ENFORCE_DISCOVERY_PRESERVE_EXISTING_CI":
        raise RoutingError("invalid existing-CI preservation mode")
    if not re.fullmatch(r"grandchallenge/[A-Za-z0-9_.-]+", config["repository"]):
        raise RoutingError("invalid repository identity")
    if os.environ.get("GITHUB_REPOSITORY") and os.environ["GITHUB_REPOSITORY"] != config["repository"]:
        raise RoutingError("repository identity mismatch")
    checks = {}
    if not isinstance(config["checks"], list) or not config["checks"]:
        raise RoutingError("empty check catalog")
    for row in config["checks"]:
        if not isinstance(row, dict) or set(row) != {"id", "test_globs"}:
            raise RoutingError("invalid check definition")
        name, globs = row["id"], row["test_globs"]
        if not isinstance(name, str) or not name or name in checks:
            raise RoutingError("duplicate/invalid check")
        if not isinstance(globs, list) or not globs or any(
            not isinstance(p, str) or not p.startswith(("tests/", "ci/", "modulus/tests/"))
            for p in globs
        ):
            raise RoutingError("invalid test discovery glob")
        checks[name] = row
    return config, checks


def _test_paths(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for directory in ("tests", "ci", "modulus/tests")
        for p in (root / directory).rglob("test_*.py")
        if p.is_file()
    }


def validate(root: Path) -> dict:
    """Validate inventory, ownership, dependency graph and executing test lanes."""
    root = root.resolve()
    config, catalog = _check_catalog(root)
    components = _components(root)
    all_tests = _test_paths(root)
    baseline = config["legacy_test_paths"]
    if not isinstance(baseline, list) or len(baseline) != len(set(baseline)):
        raise RoutingError("invalid grandfathered test inventory")
    legacy = set()
    for path in baseline:
        _relative(path)
        if path not in all_tests:
            raise RoutingError(f"stale grandfathered test path: {path}")
        legacy.add(path)
    for name, row in components.items():
        missing = set(row["checks"]) - set(catalog)
        if missing:
            raise RoutingError(f"unknown check for {name}: {sorted(missing)}")
    owned = 0
    for path in sorted(all_tests):
        name = _owner(path, components)
        if name is None:
            if path not in legacy:
                raise RoutingError(f"new test without declared owner: {path}")
            continue
        row = components[name]
        if not any(
            fnmatch.fnmatchcase(path, pattern)
            for check in row["checks"]
            for pattern in catalog[check]["test_globs"]
        ):
            raise RoutingError(f"declared test has no executing discovered check: {path}")
        owned += 1
    return {"contract_id": CONTRACT, "repository": config["repository"],
            "components": len(components), "registered_tests": owned,
            "legacy_tests": len(all_tests) - owned}


def classify(root: Path, paths: list[str]) -> dict:
    """Compute affected downstream checks; unknown changes request full catalog."""
    config, catalog = _check_catalog(root.resolve())
    components = _components(root.resolve())
    unknown, affected = [], set()
    for path in sorted(set(paths)):
        _relative(path)
        owner = _owner(path, components)
        if not owner or path.startswith(("governance/ci_components/", "governance/ci_routing_baseline.json")):
            unknown.append(path)
        else:
            affected.add(owner)
    changed = True
    while changed:
        before = len(affected)
        affected.update(name for name, row in components.items()
                        if any(upstream in affected for upstream in row["depends_on"]))
        changed = before != len(affected)
    if unknown:
        checks = sorted(catalog)
    else:
        checks = sorted({c for name in affected for c in components[name]["checks"]})
    return {"repository": config["repository"], "paths": sorted(set(paths)),
            "unknown_paths": unknown, "components": sorted(affected), "checks": checks,
            "mode": config["mode"]}


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        if not argv or argv[0] == "validate":
            print(json.dumps(validate(root), sort_keys=True))
        elif argv[0] == "classify":
            print(json.dumps(classify(root, argv[1:]), sort_keys=True))
        else:
            raise RoutingError(f"unknown command: {argv[0]}")
        return 0
    except RoutingError as exc:
        print(f"GCL-CIR-001 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
