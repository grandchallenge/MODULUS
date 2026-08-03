from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / "governance" / "regret_contract_authority.json"
POINTER_PATH = ROOT / "docs" / "standards" / "REGRET_CONTRACT_STANDARD.md"

EXPECTED_AUTHORITY = {
    "repository": "grandchallenge/gcl-standards",
    "commit": "4bb7e09cbd8ddac521447cb1386bc501f9ac5b12",
    "migration_merge": "afea7dd8e952998596fc54e215b6b1e2fcd48645",
    "chronology_cure_merge": "4bb7e09cbd8ddac521447cb1386bc501f9ac5b12",
}
EXPECTED_LOCAL_BLOBS = {
    "schemas/regret_contract.schema.json": (
        "7bf9ba77df36d1646f123c174b0116c1552bb4cd"
    ),
    "templates/regret_contract.yaml": (
        "6d0f041248d520715061bf1af8b1d97e27da0a43"
    ),
}


def git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def load_pin() -> dict[str, object]:
    value = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("authority pin must be a JSON object")
    return value


def validate() -> None:
    pin = load_pin()
    if pin.get("schema_version") != "1.0.0":
        raise ValueError("authority pin schema version drift")
    if pin.get("record_id") != "MODULUS-GCL-RC-PIN-001":
        raise ValueError("authority pin record identity drift")

    authority = pin.get("canonical_authority")
    if not isinstance(authority, dict):
        raise ValueError("canonical_authority must be an object")
    for key, expected in EXPECTED_AUTHORITY.items():
        if authority.get(key) != expected:
            raise ValueError(f"canonical authority drift: {key}")

    expected_artifacts = {
        "standard": (
            "standards/GCL-RC-00.md",
            "2978dee21e7b7ee942756b39ea49572919cf552d",
        ),
        "schema": (
            "schemas/regret_contract.schema.json",
            "7bf9ba77df36d1646f123c174b0116c1552bb4cd",
        ),
        "template": (
            "templates/regret_contract.yaml",
            "6d0f041248d520715061bf1af8b1d97e27da0a43",
        ),
        "adoption_ledger": (
            "programme-adoption/REGRET-CONTRACT-1.0.0.yaml",
            "926e213f122da621e6472ef5f5fcf9fad214fcd2",
        ),
        "decision": (
            "decisions/ADR-0002_REGRET_CONTRACT_STANDARD.md",
            "3f5eed25efe6a9b593cc07d4125b35955dc9903f",
        ),
    }
    for name, (path, blob) in expected_artifacts.items():
        artifact = authority.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"missing canonical artifact: {name}")
        if artifact.get("path") != path or artifact.get("blob") != blob:
            raise ValueError(f"canonical artifact drift: {name}")

    for relative_path, expected_blob in EXPECTED_LOCAL_BLOBS.items():
        observed = git_blob_sha(ROOT / relative_path)
        if observed != expected_blob:
            raise ValueError(f"vendored canonical copy drift: {relative_path}")

    status = pin.get("status")
    if status != {
        "standard_status": "candidate",
        "modulus_conformant": False,
        "implementation_status": "reference_candidate",
        "programme_adoption_complete": False,
    }:
        raise ValueError("status boundary drift")

    boundaries = pin.get("boundaries")
    if not isinstance(boundaries, dict) or any(boundaries.values()):
        raise ValueError("authority pin may not promote prohibited claims")

    pointer = POINTER_PATH.read_text(encoding="utf-8")
    required = (
        "grandchallenge/gcl-standards",
        "4bb7e09cbd8ddac521447cb1386bc501f9ac5b12",
        "not a normative authority",
        "`modulus.online` remains a reference implementation",
    )
    for fragment in required:
        if fragment not in pointer:
            raise ValueError(f"missing authority pointer fragment: {fragment}")
    if "Status: **Normative" in pointer:
        raise ValueError("MODULUS may not claim local normative custody")


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"regret authority pin validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("regret authority pin validation passed")
