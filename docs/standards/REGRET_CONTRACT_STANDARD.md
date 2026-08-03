# GCL Regret Contract Standard — MODULUS authority pointer

This file is **not a normative authority**.

Canonical custody of `GCL-RC-00`, Regret Contract Standard version `1.0.0`, is
held by `grandchallenge/gcl-standards` at protected commit
`4bb7e09cbd8ddac521447cb1386bc501f9ac5b12`.

Canonical identities:

- standard: `standards/GCL-RC-00.md`, blob
  `2978dee21e7b7ee942756b39ea49572919cf552d`;
- schema: `schemas/regret_contract.schema.json`, blob
  `7bf9ba77df36d1646f123c174b0116c1552bb4cd`;
- template: `templates/regret_contract.yaml`, blob
  `6d0f041248d520715061bf1af8b1d97e27da0a43`;
- adoption ledger: `programme-adoption/REGRET-CONTRACT-1.0.0.yaml`, blob
  `926e213f122da621e6472ef5f5fcf9fad214fcd2`;
- governing decision: `decisions/ADR-0002_REGRET_CONTRACT_STANDARD.md`, blob
  `3f5eed25efe6a9b593cc07d4125b35955dc9903f`.

The original MODULUS source packet is preserved at PR #1 source head
`641ba766fe8eec613a01cd4726841b1d4e93ad78`. Its normative text was migrated to
`gcl-standards`; the chronology correction was protected-merged as
`4bb7e09cbd8ddac521447cb1386bc501f9ac5b12`.

`modulus.online` remains a reference implementation. The local schema and
contract template are exact vendored copies of the canonical blobs so that
MODULUS tests and fixtures remain executable. Their presence does not transfer
standard ownership back to MODULUS.

Machine-readable custody and boundary data are recorded in
`governance/regret_contract_authority.json` and validated by
`scripts/validate_regret_authority_pin.py`.

## Current status

- standard status: `candidate`;
- MODULUS conformance: `false`;
- reference implementation status: `reference_candidate`;
- programme adoption complete: `false`.

No controller optimality, neural-network convergence, deployment safety,
mathematical certification, novelty, publication, patentability, product, or
commercial claim is established by this pointer or by the reference package.
