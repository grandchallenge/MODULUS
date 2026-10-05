# PS-DUST — Structured Perturbations for Forward-Only Credit Assignment

Status: `ACTIVE_BOUNDED_PILOT`

Authority surface: `grandchallenge/MODULUS`

Campaign identifier: `PS-DUST`

## Purpose

Determine whether geometry-aware perturbation families can reduce the forward-evaluation population required for DUST-style activation-space credit assignment, and only then test whether the surviving estimator can be coupled to MODULUS hyperspherical/tangent-coordinate updates.

The programme is intentionally falsifiable.  It is not a commitment to replace backpropagation, reproduce the full DUST pretraining result, or adopt an exact quantum parameter-shift identity.

## External source lock

Primary external implementation baseline:

- repository: `qlabs-eng/dust`
- commit: `2fdb01ba91ca38368a2b6b31a2697d77512ad175`
- `dust.py` blob: `c7d08ee9edc3a51dfbd9e73bbf787b9ca819c5c2`
- `README.md` blob: `8401c78a08bdc19ffb9741aba42558e5f0818352`

At that source lock, DUST draws independent Gaussian activation noise in `DUST.noise`, uses one-sided token-loss drops in `DUST.direct_errors`, and documents an 8-layer, width-512 transformer with supported overall populations 256, 1024, 4096, and 16384.  This campaign may reproduce or wrap that implementation for experiments, but MODULUS does not acquire authority over the external project.

## Primary hypothesis

`H1`: for an equal or smaller forward-evaluation budget, structured activation perturbations produce a credit estimate with materially higher alignment to an exact autodiff oracle than the source-locked Gaussian DUST estimator.

The primary practical target is an effective population reduction of at least `2x`.  A `4x` or greater reduction is a strong-advance result.

The null is operationally important:

`H0`: structured perturbations do not reduce the required population by at least `2x` on held-out DUST activation sites and batches after fair noise-scale calibration.

If `H0` is not rejected by the predeclared Gate 1 criteria, the structured-perturbation line stops before training-scale work.

## Perturbation families

The initial comparison set is deliberately small:

1. `gaussian_one_sided` — source-compatible control.
2. `gaussian_antithetic` — symmetric finite-difference control.
3. `rademacher_one_sided` — discrete isotropic control.
4. `hadamard_one_sided` — randomized orthogonal basis rows.
5. `hadamard_antithetic` — randomized orthogonal basis rows with symmetric differences.

The Hadamard family is especially natural for the source-locked width `512`: a full block spans one orthogonal basis without requiring dense QR construction.

## Metrics

Every estimator comparison reports, at minimum:

- perturbation family;
- antithetic status;
- activation site and layer;
- batch/checkpoint identity;
- noise scale `sigma`;
- perturbation population `K`;
- total forward evaluations attributable to the estimate;
- cosine similarity to the exact reference gradient;
- relative L2 error;
- estimator variance across independent randomized bases/seeds;
- wall-clock and peak-memory diagnostics when measured on accelerator hardware.

Forward-evaluation count, not nominal population alone, is the primary compute ledger.  Antithetic methods therefore pay for both signs.

## Calibration and holdout rule

Noise-scale choice must not be tuned on the final evaluation rows.

For Gate 1:

1. choose a small fixed calibration set of activation sites/batches;
2. sweep a predeclared sigma grid on that calibration set;
3. freeze one sigma per estimator family, or one common sigma if the calibration evidence supports it;
4. score the frozen choices on held-out sites/batches;
5. report every attempted calibration value, not only the selected value.

Suggested first sigma grid: `{0.005, 0.01, 0.02, 0.05}`.

## Work-package ladder

### PS-DUST-WP00 — estimator implementation and deterministic sanity fixture

Purpose: validate the perturbation generators, evaluation accounting, and gradient-estimator implementation independently of DUST.

Artifacts:

- `modulus/estimators/structured_zeroth_order.py`;
- estimator unit tests;
- `scripts/run_ps_dust_gate0.py`;
- retained deterministic fixture `experiments/ps_dust/PS_DUST_GATE0_001.json`.

Acceptance:

- CI tests and lint pass;
- full randomized Hadamard basis recovers a linear gradient to numerical tolerance under antithetic evaluation;
- deterministic nonlinear fixture is retained with an explicit non-claim boundary.

Scientific effect: `NONE`.  Passing WP00 only authorizes Gate 1 execution.

### PS-DUST-WP01 — source-locked DUST activation-gradient alignment

Purpose: answer the campaign's primary question without training.

Required comparison:

- exact same frozen model, batch, activation site, and loss;
- exact autodiff gradient used only as evaluation oracle;
- source-compatible Gaussian one-sided estimator reproduced first;
- structured estimators then evaluated under the same site/batch conditions;
- populations drawn from `{16, 32, 64, 128, 256, 512}` where valid;
- compute-equivalent comparisons use forward-evaluation count, not merely equal `K`.

Initial site coverage:

- writer/output-projection activation sites across early, middle, and late blocks;
- at least one MLP hidden/output site;
- any source-specific site whose semantics differ materially is reported separately rather than pooled silently.

Checkpoint coverage:

- initialization fixture;
- at least one nontrivial trained checkpoint generated under a documented source-compatible path before promotion beyond Gate 1.

Primary advance criterion:

On held-out rows, a structured method must reach the reference Gaussian estimator's target cosine at no more than half its forward-evaluation budget, without materially worse relative L2 error.

Strong advance criterion:

At least `4x` lower forward-evaluation budget at matched target cosine on a majority of held-out site/batch strata.

Kill criterion:

If no structured family demonstrates at least `2x` effective population/evaluation reduction after calibration and held-out evaluation, close the campaign as a useful negative result.  Do not proceed to WP02 or WP03.

### PS-DUST-WP02 — short matched-compute learning test

Gate: WP01 advance criterion must pass.

Purpose: determine whether an estimator-alignment gain survives optimization dynamics.

Compare source-compatible Gaussian DUST against the best frozen structured estimator at matched forward-compute budgets.  The decisive comparison is reduced structured population at similar validation loss, not equal-population leaderboard performance.

Advance criterion:

A structured configuration reaches statistically compatible validation loss with no more than half the Gaussian forward-evaluation budget, without an offsetting material memory or wall-clock regression.

### PS-DUST-WP03 — structured tangent-coordinate coupling

Gate: WP02 passes.

Purpose: connect DUST-style global activation credit to MODULUS geometry.

Candidate local parameterization:

`h' = R(theta) h`,

where `R(theta)` is composed from Givens/orthogonal transforms.  DUST-style activation credit estimates the error signal at `h'`; local structured derivatives map that signal into a low-dimensional tangent coordinate update.

This WP must compare against a parameter-count-matched conventional adapter and against the best WP02 structured activation estimator without tangent-coordinate coupling.

### PS-DUST-WP04 — forward-only adaptation demonstration

Gate: WP03 yields a reproducible advantage.

Purpose: test the most commercially plausible use case before any large-scale pretraining claim: frozen-model adaptation with small norm-preserving structured adapters and no end-to-end backward pass through the base model.

Required outputs include end-task utility, memory, wall-clock, forward-evaluation count, adapter parameter count, and conventional LoRA/backprop controls.

## Claim firewall

The following do not follow from WP00 or a successful WP01:

- that DUST is more compute-efficient than backpropagation;
- that structured perturbations improve full transformer training;
- that an exact quantum parameter-shift identity applies to the transformer loss;
- that a synthetic fixture demonstrates transformer-scale variance reduction;
- that an orthogonal basis is optimal;
- that a forward-only method is commercially superior.

Each later claim requires its own admitted experiment.

## Why this belongs in MODULUS

MODULUS already treats geometry as a first-class optimizer control surface, supports hyperspherical/Hyperball dynamics, tangent projection, orthogonal LoRA steering, and reproducible ablation tooling.  PS-DUST extends that remit from geometric treatment of backpropagated updates to the geometry of forward-only credit estimation.  It is therefore execution-adjacent to MODULUS without changing MODULUS into a general training framework.

## Immediate next action

Complete WP00 through protected CI, retain the deterministic fixture, then execute WP01 against the exact DUST source lock.  No training-scale allocation is authorized until the WP01 kill/advance decision is recorded.
