# PS-DUST-WP01-A — Initialization estimator screen result

Status: `STOPPED_AT_GATE1`

Tracker: `grandchallenge/MODULUS#26`

Authority boundary: bounded initialization-only estimator evidence. This result does not
establish transformer-training efficiency, scaling, novelty, deployment value, commercial
superiority, or superiority to backpropagation.

## Protected execution

External source:

- `qlabs-eng/dust@2fdb01ba91ca38368a2b6b31a2697d77512ad175`;
- `dust.py` blob `c7d08ee9edc3a51dfbd9e73bbf787b9ca819c5c2`;
- `model.py` blob `e605727f6ae9bb114647ae04308d98d0ba5b9dee`.

Protocol:

- `PS_DUST_WP01_PROTOCOL.md`;
- amendments A001, A002, A003, A004.

Runtime:

- NVIDIA GeForce RTX 2080;
- PyTorch `2.11.0+cu130`;
- source `nrep=2`;
- model seed `42`;
- sequence length `64`.

## Calibration receipt

Artifact: `PS_DUST_WP01_A_CALIBRATION_001.json`

Rows: `192`

Elapsed: `80.283324275 s`

SHA-256:

`419d49c0f892e7d2ae91e62a16036d22bb30f7fe59869863dde28a81b473b0ee`

Mechanically frozen held-out sigma map:

| Variant | Sigma |
|---|---:|
| gaussian_independent | 0.20 |
| gaussian_paired_chunk | 0.20 |
| hadamard_independent | 0.20 |
| hadamard_paired_chunk | 0.40 |
| rademacher_independent | 0.20 |
| rademacher_paired_chunk | 0.40 |

The Gaussian-independent source control remained fixed at the source-compatible `0.20`
by protocol rather than by outcome selection.

## Held-out receipt

Artifact: `PS_DUST_WP01_A_HELDOUT_001.json`

Rows: `1152`

Elapsed: `250.009252272 s`

SHA-256:

`04b55c7c59f02cef4f39bd5d1e672033fb7eb4777e586169c9ce68429d3ead6d`

The source Gaussian baseline reached a registered target only for cosine `0.50` on the
three held-out writer sites. In all three baseline-reachable strata its first crossing was
at `K=512`. The `0.70` and `0.85` targets and the held-out MLP stratum were
baseline-censored within the registered K grid.

## Candidate decision summary

| Candidate | Median R | Median relative-L2 ratio | L2 safeguard | Sites with R >= 2 | Advance |
|---|---:|---:|---|---:|---|
| rademacher_independent | 1.0 | 0.9988057191 | PASS | 0 | NO |
| hadamard_independent | 1.0 | 0.8190942883 | PASS | 0 | NO |
| gaussian_paired_chunk | 1.0 | 1.6116927805 | FAIL | 0 | NO |
| rademacher_paired_chunk | 1.0 | 1.6096065029 | FAIL | 0 | NO |
| hadamard_paired_chunk | 1.0 | 1.1507820751 | FAIL | 0 | NO |

No registered candidate reduced the first-crossing draw budget below `K=512` on any
baseline-reachable writer stratum. Therefore no candidate achieved the preregistered
`>=2x` population-efficiency criterion on even one held-out site, much less two.

## Useful negative evidence

The strongest non-advancing signal is `hadamard_independent`.

At the matched `K=512` crossing it preserved `R=1.0` but reduced median relative-L2
against the same-token reference. The aggregate candidate summary gives a median
relative-L2 ratio of approximately `0.819` versus Gaussian, satisfying the quality
safeguard but not the compute-efficiency criterion.

This is evidence that orthogonal coverage can improve estimate quality at a fixed large
budget on this initialization fixture. It is **not** evidence of the required population
reduction: the improvement was insufficient to move the first target crossing down one
registered factor-of-two K rung.

The paired-chunk variants did not rescue the hypothesis. Their median population-efficiency
ratio remained `1.0`; all three paired variants also failed or approached the declared
relative-L2 safeguard, with Gaussian and Rademacher paired variants materially worse.

## Disposition

The preregistered WP01-A stop rule fires:

`STOPPED_AT_GATE1`

Consequences:

1. Do not execute WP01-B trained-checkpoint confirmation.
2. Do not execute WP02 matched-compute training.
3. Do not execute WP03 tangent-coordinate coupling under this campaign hypothesis.
4. Preserve the Hadamard fixed-budget quality signal as bounded negative/diagnostic evidence.
5. Any future attempt to exploit that signal requires a new registered hypothesis rather
   than silently weakening the `>=2x` criterion.

The campaign has therefore answered its practical first question: under this source-locked
initialization screen and registered forward-budget grid, structured perturbations did not
buy the factor-of-two reduction required to justify additional DUST training compute.
