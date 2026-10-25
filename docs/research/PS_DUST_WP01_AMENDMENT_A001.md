# PS-DUST-WP01 Amendment A001 — source-compatible sigma calibration

Status: `ADOPTED_PRE_OUTCOME`

Recorded before any Gate-1 estimator outcome was inspected.

## Reason

At the locked external source `qlabs-eng/dust@2fdb01ba91ca38368a2b6b31a2697d77512ad175`, `DUST.step` calls `direct_errors(..., sigma=0.2)` for writer/MLP/embedding sites. It uses `sigma=0.2` for early `o@{layer}` attention-output sites and `sigma=0.4` for the deeper half.

The original provisional WP01 calibration grid `{0.005, 0.01, 0.02, 0.05}` omitted the source operating point and therefore could not fairly reproduce the Gaussian control.

## Amendment

- writer/MLP direct-site calibration grid: `{0.05, 0.10, 0.20, 0.40}`;
- `o@` attention-output calibration grid: `{0.10, 0.20, 0.40, 0.80}`;
- the source-compatible Gaussian control must include the exact source sigma for its site class;
- structured families use the same declared grid within a site class;
- the original grid remains part of repository history and is superseded for WP01 by this amendment.

## Unchanged

The population grid, held-out rule, metrics, `>=2x` primary advance threshold, `>=4x` strong-advance threshold, stop rule, and claim firewall are unchanged.

## Source evidence

- `dust.py` source blob: `c7d08ee9edc3a51dfbd9e73bbf787b9ca819c5c2`;
- source commit: `2fdb01ba91ca38368a2b6b31a2697d77512ad175`.
