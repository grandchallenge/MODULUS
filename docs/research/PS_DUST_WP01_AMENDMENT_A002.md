# PS-DUST-WP01 Amendment A002 — compute ledger

Status: `ADOPTED_PRE_OUTCOME`

Repository note: this file is a post-outcome transcription of the pre-outcome amendment
recorded on `grandchallenge/MODULUS#26` before Gate-1 calibration or held-out execution.
The authoritative historical issue comment is
`#issuecomment-6006015996`. This transcription changes no experiment setting or result.

## Reason

Source DUST uses `nrep=2` at population 256, so two perturbation draws are vectorized
inside one model invocation. Vectorization must not be confused with estimator efficiency.

## Required compute ledger

Every scientific row records both:

- `physical_forward_calls`: actual perturbation model invocations;
- `draw_equivalent_forward_evaluations`: perturbation members evaluated, normalized to
  one clean-batch forward each.

The preregistered efficiency ratio `R` uses draw-equivalent forward evaluations.

Physical calls, elapsed time, and peak memory are secondary implementation diagnostics.

For source one-sided direct errors with `nrep=2`:

- `K` draws imply `K/2` physical perturbation model calls;
- `K` draw-equivalent evaluations are charged.

For the separately registered antithetic extension:

- `K` unique directions evaluated at both signs imply `K` physical perturbation calls
  under `nrep=2`;
- `2K` draw-equivalent evaluations are charged.

No population, sigma, holdout, metric, `>=2x` advance, `>=4x` strong-advance, or
stop threshold was changed by this amendment.
