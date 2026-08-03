# MODULUS Documentation Hub

This is the entry point for technical, contextual, and operational documentation.

## 1) API Reference (Auto-Generated)

- API index: [api_generated/INDEX.md](./api_generated/INDEX.md)
- Source of truth: code introspection via `scripts/generate_api_docs.py`

## 2) Context and Motivation

- [context/MOTIVATION_AND_CONTEXT.md](./context/MOTIVATION_AND_CONTEXT.md)

## 3) Pedagogy and Learning Path

- [pedagogy/LEARNING_PATH.md](./pedagogy/LEARNING_PATH.md)
- [../notebooks/MODULUS_Pedagogical_Walkthrough.ipynb](../notebooks/MODULUS_Pedagogical_Walkthrough.ipynb)

## 4) Future Extensions

- [roadmap/FUTURE_EXTENSIONS.md](./roadmap/FUTURE_EXTENSIONS.md)

## 5) Integration and Engineering Operations

- [integration_guide.md](./integration_guide.md)
- [engineering/INSTALL.md](./engineering/INSTALL.md)
- [engineering/ENGINEERING_HANDOFF_GCT_2026-03-04.md](./engineering/ENGINEERING_HANDOFF_GCT_2026-03-04.md)
- [engineering/LINT_PHASE_PLAN.md](./engineering/LINT_PHASE_PLAN.md)

## 6) Standards and Adaptive Control

- Canonical standard authority: `grandchallenge/gcl-standards` at
  `4bb7e09cbd8ddac521447cb1386bc501f9ac5b12`
- [MODULUS authority pointer](./standards/REGRET_CONTRACT_STANDARD.md)
- [Online-control rollout](./standards/ONLINE_CONTROL_ROLLOUT.md)
- Machine authority pin:
  [../governance/regret_contract_authority.json](../governance/regret_contract_authority.json)
- Vendored canonical schema:
  [../schemas/regret_contract.schema.json](../schemas/regret_contract.schema.json)
- Vendored canonical template:
  [../templates/regret_contract.yaml](../templates/regret_contract.yaml)
- Reference implementation: `modulus.online`

The local pointer, schema, template, and implementation do not transfer
normative standard custody to MODULUS or establish programme conformance.

## 7) Compliance and Governance

- [ip/provenance.md](./ip/provenance.md)
- [ip/ai_origin_evidence.md](./ip/ai_origin_evidence.md)
- [ip/SIGNOFF.md](./ip/SIGNOFF.md)

## Auto-Update Workflow

Generate API docs:

```bash
python scripts/generate_api_docs.py
```

Check freshness (CI mode):

```bash
python scripts/generate_api_docs.py --check
```

Validate the Regret Contract authority pin:

```bash
python scripts/validate_regret_authority_pin.py
```
