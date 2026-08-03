from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_regret_authority_pin",
    ROOT / "scripts" / "validate_regret_authority_pin.py",
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _pin() -> dict[str, object]:
    return json.loads(
        (ROOT / "governance" / "regret_contract_authority.json").read_text(
            encoding="utf-8"
        )
    )


def test_authority_pin_validates() -> None:
    validator.validate()


def test_commit_drift_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = copy.deepcopy(_pin())
    broken["canonical_authority"]["commit"] = "0" * 40
    monkeypatch.setattr(validator, "load_pin", lambda: broken)
    with pytest.raises(ValueError, match="canonical authority drift"):
        validator.validate()


def test_programme_conformance_inflation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = copy.deepcopy(_pin())
    broken["status"]["modulus_conformant"] = True
    monkeypatch.setattr(validator, "load_pin", lambda: broken)
    with pytest.raises(ValueError, match="status boundary drift"):
        validator.validate()


def test_claim_inflation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = copy.deepcopy(_pin())
    broken["boundaries"]["controller_optimality_claimed"] = True
    monkeypatch.setattr(validator, "load_pin", lambda: broken)
    with pytest.raises(ValueError, match="prohibited claims"):
        validator.validate()
