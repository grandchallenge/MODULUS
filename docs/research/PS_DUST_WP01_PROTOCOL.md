# PS-DUST-WP01 — DUST activation-gradient alignment protocol

Status: `READY_AFTER_WP00_INTEGRATION`

Campaign: `PS-DUST`

## Objective

Measure whether structured activation perturbations reduce the forward-evaluation budget needed to approximate a source-locked DUST credit signal's exact autodiff reference direction.

This is an estimator experiment, not a training experiment.

## Source boundary

Use `qlabs-eng/dust@2fdb01ba91ca38368a2b6b31a2697d77512ad175` as the external implementation baseline.  Record any compatibility patch as a separate local artifact; do not silently modify the baseline semantics.

## Required controls

1. Reproduce `DUST.direct_errors` Gaussian one-sided behavior before adding new perturbation families.
2. Hold model parameters, data batch, activation site, and loss fixed within each comparison cell.
3. Use exact autodiff only to construct the reference gradient for evaluation.
4. Report nominal perturbation population and actual forward evaluations separately.
5. Keep calibration and held-out rows disjoint.
6. Preserve failed or unfavorable rows.

## Initial matrix

Populations: `16, 32, 64, 128, 256, 512`.

Sigma calibration grid: `0.005, 0.01, 0.02, 0.05`.

Families:

- Gaussian one-sided;
- Gaussian antithetic;
- Rademacher one-sided;
- randomized Hadamard one-sided;
- randomized Hadamard antithetic.

Seeds/randomized bases: at least `8` per held-out cell unless accelerator cost forces an explicitly recorded amendment before outcomes are inspected.

Layer strata:

- early;
- middle;
- late.

At minimum, include writer/output-projection sites and one MLP site.  Do not aggregate semantically different DUST site classes without preserving per-class rows.

## Reference-gradient semantics

The implementation must document exactly which scalar loss the autodiff oracle differentiates and which activation tensor is differentiated.  If source DUST credit intentionally omits cross-token or downstream effects that appear in the total-loss gradient, report both semantics separately rather than calling either one "the" gradient.

A semantic mismatch between source-local credit and total-loss backprop is itself a result and must not be hidden by pooling.

## Primary decision statistic

For each held-out stratum, form the lowest forward-evaluation budget at which each estimator reaches a predeclared target cosine.  Initial target cosines are `0.50`, `0.70`, and `0.85`; strata that cannot reach a target are reported as censored rather than extrapolated.

Define the population-efficiency ratio against source Gaussian as:

`R = evaluations_gaussian / evaluations_structured`

at matched target cosine and stratum.

Primary campaign advance requires median held-out `R >= 2` with no material relative-L2 degradation at the matched target.

Strong advance requires median `R >= 4` and a majority of held-out strata individually exceeding `4x`.

## Required artifact schema

Every row must include:

- source commit and local adapter commit;
- model/checkpoint identity;
- data split/batch identity;
- site name, layer, tensor shape, and site semantic class;
- estimator family and antithetic flag;
- sigma;
- population;
- forward evaluations;
- seed/basis identity;
- cosine similarity;
- relative L2 error;
- reference norm and estimate norm;
- elapsed time if measured;
- peak device memory if measured;
- calibration/held-out flag.

## Stop rule

If the held-out evaluation fails the `2x` primary advance criterion, record `STOPPED_AT_GATE1` and close the structured-perturbation expansion.  A negative result remains durable evidence and should not be converted into an unregistered estimator search.
