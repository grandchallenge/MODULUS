# GCL-GHOS-MODULUS-POST-REPAIR-READBACK-COLLECTOR-001

This record attests the closed collector implementation for `grandchallenge/MODULUS#13`.

The collector is GET-only. It requires the caller to provide the exact current protected `main` SHA and proves that SHA is identical to or descends from admitted owner-controls evidence merge `f54dd2c0b26ea46ef6b598f6a65dfcef2c47da47` with that merge as the comparison base and merge base.

Permanent implementation identities:

- collector SHA-256: `182b33310c09ee7951e80df368d8666be8129b6c4d57ec0ced1b3af0e0cb5592`;
- schema SHA-256: `186c53172c835e688fddeb4fdfb56364d15dff7504beefa69b00404066ce010f`;
- mutation-test SHA-256: `417a5ef76ea49dcfb56a3ce25db9aff43354093590ea376f1c483fc55ab46b3a`;
- schema version: `1.1.0`;
- deterministic local mutation tests: 16/16 passed;
- closed circular-head cure run `30862735921`: success.

The collector requires repository-admin proof, a stable caller-supplied protected-main identity, ancestry proof from the admitted evidence merge, exact settings, exact list/detail equality for rulesets `20266757` and `20334249`, enabled security controls, active workflow inventory, six successful protected-main contexts, and exact governed surface identities and contents.

It rejects missing or malformed target identities, main-head movement, ancestry or merge-base drift, authority drift, settings drift, ruleset drift, failed checks, missing workflows or surfaces, mutable action references, credential-like material, claim promotion, and any attempt to authorize MODULUS deviation disposition.

The reconstruction payload, bounded transfer chunks, and temporary cure workflows are absent from the final diff.

No live readback or setting mutation is included. `MODULUS-P1-001`, `MODULUS-P2-001`, MODULUS conformance, and organization-wide conformance remain unresolved.
