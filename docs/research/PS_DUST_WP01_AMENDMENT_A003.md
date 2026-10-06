# PS-DUST-WP01 Amendment A003 — in-chunk paired directions

Status: `ADOPTED_PRE_OUTCOME`

Repository note: this file is a post-outcome transcription of the pre-outcome estimator
registration recorded on `grandchallenge/MODULUS#26` before Gate-1 calibration or
held-out execution. The authoritative historical issue comment is
`#issuecomment-6006055615`. This transcription changes no experiment setting or result.

## Rationale

At source population 256, DUST has `nrep=2` and centers each token reward across the two
perturbation members in the draw chunk.

The campaign had already registered symmetric `+a/-a` perturbations. The initial
engineering implementation evaluated the two signs in separate calls. A source-native
paired form can instead place `+a` and `-a` in the same `nrep=2` chunk and pass them
through the unmodified source `DUST.direct_errors` path.

## Registered variants

- `gaussian_paired_chunk`;
- `rademacher_paired_chunk`;
- `hadamard_paired_chunk`.

For `K` total perturbation draws, these variants use `K/2` unique directions arranged
as consecutive `(+a,-a)` pairs.

Source reward centering therefore yields the symmetric directional signal within the
existing vectorized call.

## Compute accounting

For the paired-chunk variants:

- `K` draw-equivalent evaluations;
- `K/2` physical perturbation model calls at source `nrep=2`;
- the same clean-capture accounting as the Gaussian-independent control.

The separate-call antithetic implementation remains an engineering/control variant but is
not privileged in the primary efficiency comparison.

No Gate-1 threshold, holdout partition, or stop rule was changed by this registration.
