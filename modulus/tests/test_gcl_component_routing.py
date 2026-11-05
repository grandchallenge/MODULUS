"""Adversarial tests for the GCL-CIR-001 portable CI ownership contract."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ci/gcl_component_routing.py"
spec = importlib.util.spec_from_file_location("gcl_ci_routing_standard", SOURCE)
router = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = router
spec.loader.exec_module(router)


def component(
    name="alpha", *, deps=(), checks=("ci",), exact=(), test_prefixes=("tests/test_alpha_",)
):
    return {
        "contract_id": "GCL-CIR-001",
        "version": "1.0.0",
        "component_id": name,
        "owner": "Adversarial test component",
        "paths": {"exact": list(exact), "prefixes": [f"research/{name}/"],
                  "test_prefixes": list(test_prefixes)},
        "depends_on": list(deps),
        "checks": list(checks),
    }


class RoutingStandardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "governance/ci_components").mkdir(parents=True)
        (self.root / "tests").mkdir()
        self.config = {
            "contract_id": "GCL-CIR-001", "version": "1.0.0",
            "repository": "grandchallenge/MODULUS",
            "mode": "ENFORCE_DISCOVERY_PRESERVE_EXISTING_CI",
            "checks": [{"id": "ci", "test_globs": ["tests/test_*.py"]}],
            "legacy_test_paths": [],
        }
        self.write_config()
        self.write_component(component())

    def write_config(self):
        (self.root / "governance/ci_routing_baseline.json").write_text(
            json.dumps(self.config), encoding="utf-8")

    def write_component(self, row):
        (self.root / f"governance/ci_components/{row['component_id']}.json").write_text(
            json.dumps(row), encoding="utf-8")

    def test_real_repository_is_registered(self):
        result = router.validate(ROOT)
        self.assertEqual(result["repository"], "grandchallenge/MODULUS")
        self.assertGreaterEqual(result["registered_tests"], 1)

    def test_discovered_test_with_executing_check(self):
        (self.root / "tests/test_alpha_new.py").write_text("pass
")
        result = router.validate(self.root)
        self.assertEqual(result["registered_tests"], 1)
        self.assertEqual(router.classify(self.root, ["tests/test_alpha_new.py"])["checks"], ["ci"])

    def test_new_unowned_test_rejected_without_baseline_change(self):
        (self.root / "tests/test_other_new.py").write_text("pass
")
        with self.assertRaisesRegex(router.RoutingError, "new test without declared owner"):
            router.validate(self.root)
        self.config["legacy_test_paths"].append("tests/test_other_new.py")
        self.write_config()
        self.assertEqual(router.validate(self.root)["legacy_tests"], 1)

    def test_missing_runner_rejected(self):
        row = component(checks=("unavailable",))
        self.write_component(row)
        with self.assertRaisesRegex(router.RoutingError, "unknown check"):
            router.validate(self.root)
        row = component(checks=("ci",), test_prefixes=("ci/test_alpha_",))
        self.write_component(row)
        (self.root / "ci").mkdir()
        (self.root / "ci/test_alpha_new.py").write_text("pass
")
        with self.assertRaisesRegex(router.RoutingError, "no executing discovered check"):
            router.validate(self.root)

    def test_dependency_closure_and_unknown_fallback(self):
        self.write_component(
            component("beta", deps=("alpha",), test_prefixes=("tests/test_beta_",))
        )
        routed = router.classify(self.root, ["research/alpha/spec.json"])
        self.assertEqual(routed["components"], ["alpha", "beta"])
        self.assertEqual(routed["checks"], ["ci"])
        fallback = router.classify(self.root, ["new/proposal.txt"])
        self.assertEqual(fallback["unknown_paths"], ["new/proposal.txt"])
        self.assertEqual(fallback["checks"], ["ci"])

    def test_ambiguous_owner_rejected(self):
        self.write_component(component("beta", test_prefixes=("tests/test_alpha_",)))
        with self.assertRaisesRegex(router.RoutingError, "ambiguous ownership"):
            router.classify(self.root, ["tests/test_alpha_new.py"])

    def test_missing_dependency_and_cycle_rejected(self):
        self.write_component(component("alpha", deps=("nonexistent",)))
        with self.assertRaisesRegex(router.RoutingError, "unknown dependency"):
            router.validate(self.root)
        self.write_component(component("alpha", deps=("beta",)))
        self.write_component(
            component("beta", deps=("alpha",), test_prefixes=("tests/test_beta_",))
        )
        with self.assertRaisesRegex(router.RoutingError, "dependency cycle"):
            router.validate(self.root)

    def test_control_plane_and_unsafe_prefix_rejected(self):
        row = component()
        row["paths"]["prefixes"] = ["ci/"]
        self.write_component(row)
        with self.assertRaisesRegex(router.RoutingError, "captured protected control path"):
            router.validate(self.root)
        row["paths"]["prefixes"] = ["research/../unsafe/"]
        self.write_component(row)
        with self.assertRaisesRegex(router.RoutingError, "unsafe path"):
            router.validate(self.root)

    def test_stale_legacy_inventory_rejected(self):
        self.config["legacy_test_paths"] = ["tests/test_deleted.py"]
        self.write_config()
        with self.assertRaisesRegex(router.RoutingError, "stale grandfathered test path"):
            router.validate(self.root)


if __name__ == "__main__":
    unittest.main()
