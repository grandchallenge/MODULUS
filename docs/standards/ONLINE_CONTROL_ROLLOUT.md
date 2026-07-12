# GCL Online-Control Rollout

## Objective

Operationalize the Regret Contract Standard without destabilizing existing
training or routing systems. The first release is an outer-loop governor over
small action spaces. Replacement of AdamW, Muon, MODULUS Hyperball, or existing
AETHER execution semantics is explicitly out of scope.

## Work packages

### WP-RC-01 — Canonical standard and reference library

Owner: MODULUS  
Artifacts: schema, template, contract dataclasses, OMD/Hedge/Exp3 primitives,
regret telemetry, tests, deterministic fixture runner.  
Exit: CI green and fixture JSON retained as an artifact.

### WP-RC-02 — KIBO optimistic governor

Implement a groupwise learning-rate multiplier controller. Compare zero-hint OMD
with Koopman-hinted OMD. Log dual-norm hint error, path length of the hindsight
comparator, boundary gain, and fallback activations.

Required run matrix:

1. fixed learning rate;
2. tuned fixed group multipliers;
3. zero-hint OMD;
4. persistence-hint OMD;
5. Koopman-hint OMD;
6. deliberately misspecified hints.

### WP-RC-03 — AETHER sleeping-bandit router

Represent eligibility as an availability mask produced by AETHER-POL governance.
Use cost-normalized bounded loss. Compare static routing, full-information Hedge
where feasible, and sleeping Exp3 under delayed outcomes. Store each action,
probability, estimate, and outcome in typed tuples.

### WP-RC-04 — SPINDLE dynamic scheduler

Treat operator orderings/splittings as experts. Measure task loss, lifted
commutator, boundary amplification, router churn, and compute. Evaluate one
controlled phase switch and one gradual drift fixture. Report static and
one-switch tracking regret.

### WP-RC-05 — Tricorder anytime monitoring

Add bounded-observable confidence sequences for repeatedly inspected metrics.
Initial observables: orthogonality error, boundary gain, eigengap proxy, and
router concentration. Document the stochastic assumptions for each sequence;
do not apply the confidence sequence to unbounded raw statistics.

### WP-RC-06 — Groupwise adaptive-beta pilot

Control beta/momentum through either a small finite candidate set or bounded
multipliers. Counterfactual loss provenance must distinguish shadow moments,
local approximation, and periodic replay. Compare against AdamW, AdaBelief, and
the existing CA-AdamW/TEMA-AdamW specification.

## Common report fields

Every work package reports:

- contract identifier and schema version;
- base system and fallback;
- action dimension and update frequency;
- feedback type and delay distribution;
- loss decomposition and bounds;
- comparator construction and provenance;
- cumulative, static, primary, and interval regret;
- task metrics, stability metrics, tokens/s, memory, and controller overhead;
- projection, abstention, and rollback counts;
- open assumptions and failed fixtures.

## Promotion rule

A controller progresses from fixture to optional experiment flag only after it
beats or matches the stable base on end-task utility and stability within the
declared compute budget. It becomes a default only after a second modality or
workload reproduces the benefit. Novelty is not a promotion criterion.
