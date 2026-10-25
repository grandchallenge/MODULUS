# PS-DUST-WP01 Amendment A004 — initialization screen execution freeze

Status: `ADOPTED_PRE_OUTCOME`

Tracker: `grandchallenge/MODULUS#26`

This amendment was recorded on the tracker before any WP01-A calibration or held-out
scientific row was executed. It partitions the initialization screen exactly and does not
change the existing `>=2x` advance or `<2x` stop rule.

## Model and data

- source: `qlabs-eng/dust@2fdb01ba91ca38368a2b6b31a2697d77512ad175`;
- model seed: `42`;
- checkpoint: initialization only;
- sequence length: `64`;
- calibration token batch seed: `314159`;
- held-out token batch seed: `271828`;
- sampled token positions: eight deterministic linspace positions from `0` through `63`.

Estimator-seed replicas use the same frozen token sequence inside each comparison cell and
are vectorized through the ordinary model batch dimension only to reduce wall-clock cost.
The source transformer has no cross-example operation, so this changes execution packing,
not the per-replica estimator definition. Physical-call timing and memory remain secondary
diagnostics; draw-equivalent forward evaluations remain the primary compute ledger.

## Calibration partition

Calibration sites, excluded from held-out decision rows:

- `transformer.h.1.attn.c_proj`;
- `transformer.h.1.mlp.c_fc`.

Calibration estimator seeds: `2000, 2001`.

Calibration draw budgets: `K in {64, 128}`.

Sigma grid: `{0.05, 0.10, 0.20, 0.40}`.

The source Gaussian-independent control is frozen to `sigma=0.20` for held-out
evaluation regardless of calibration ranking. Every other registered variant selects one
sigma by:

1. highest mean same-token cosine across all calibration site/K/seed cells;
2. lower mean relative-L2 as first tie-break;
3. smaller sigma as second tie-break.

## Held-out partition

Held-out sites:

- early writer: `transformer.h.0.attn.c_proj`;
- middle writer: `transformer.h.4.attn.c_proj`;
- late writer: `transformer.h.7.attn.c_proj`;
- MLP: `transformer.h.4.mlp.c_fc`.

Held-out estimator seeds: `3000..3007`.

Held-out draw grid: `K in {16, 32, 64, 128, 256, 512}`.

Primary reference semantics: same-token/local autodiff reference.

The total-loss autodiff reference remains separately reported diagnostic evidence and is
not pooled into the primary decision.

## Registered primary variants

- `gaussian_independent` — source control;
- `rademacher_independent`;
- `hadamard_independent`;
- `gaussian_paired_chunk`;
- `rademacher_paired_chunk`;
- `hadamard_paired_chunk`.

A003 paired-chunk variants use `K` total perturbation draws, comprising `K/2`
unique `(+a,-a)` direction pairs inside the unmodified source
`DUST.direct_errors` path at `nrep=2`. Independent variants use `K` unique draws.

Every scientific row records:

- `K` draw-equivalent forward evaluations;
- `K/2` physical perturbation model calls under source `nrep=2`;
- clean capture separately from the perturbation ledger.

## Decision statistic

Target cosines: `{0.50, 0.70, 0.85}`.

For each held-out site × target and each estimator, use the lowest registered `K` whose
median across the eight frozen estimator seeds reaches target.

- If Gaussian does not reach target, the comparator stratum is baseline-censored and no
  efficiency ratio is defined.
- If Gaussian reaches target but a candidate does not, candidate `R=0` for that stratum;
  candidate failures are not silently dropped.
- Otherwise `R = K_gaussian / K_candidate`.

For each registered candidate, report median `R` over all baseline-reachable site ×
target strata.

Relative-L2 safeguard: at matched crossing rows, candidate median relative-L2 must be no
more than `1.10x` the Gaussian median relative-L2.

Primary advance requires at least one registered candidate with:

- median `R >= 2`;
- the relative-L2 safeguard satisfied;
- `R >= 2` on at least two distinct held-out sites.

Strong advance retains the existing `>=4x` criterion.

If no candidate satisfies primary advance, disposition is `STOPPED_AT_GATE1`; WP01-B
trained-checkpoint confirmation is not executed.
