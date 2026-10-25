#!/usr/bin/env python3
"""Execute the preregistered PS-DUST-WP01-A initialization estimator screen.

Protocol authority:
- docs/research/PS_DUST_WP01_PROTOCOL.md
- issue #26 amendments A001-A004

The script separates calibration and held-out execution. Held-out execution reads
the frozen sigma choices from a calibration artifact; it does not re-tune them.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Iterator

import torch

DUST_COMMIT = "2fdb01ba91ca38368a2b6b31a2697d77512ad175"
DUST_BLOBS = {
    "dust.py": "c7d08ee9edc3a51dfbd9e73bbf787b9ca819c5c2",
    "model.py": "e605727f6ae9bb114647ae04308d98d0ba5b9dee",
}
MODEL_SEED = 42
SEQUENCE_LENGTH = 64
SAMPLED_TOKENS = 8

CALIBRATION_BATCH_SEED = 314159
HELDOUT_BATCH_SEED = 271828
CALIBRATION_ESTIMATOR_SEEDS = [2000, 2001]
HELDOUT_ESTIMATOR_SEEDS = list(range(3000, 3008))

CALIBRATION_SITES = [
    "transformer.h.1.attn.c_proj",
    "transformer.h.1.mlp.c_fc",
]
HELDOUT_SITES = [
    "transformer.h.0.attn.c_proj",
    "transformer.h.4.attn.c_proj",
    "transformer.h.7.attn.c_proj",
    "transformer.h.4.mlp.c_fc",
]
CALIBRATION_K = [64, 128]
HELDOUT_K = [16, 32, 64, 128, 256, 512]
SIGMA_GRID = [0.05, 0.10, 0.20, 0.40]
TARGET_COSINES = [0.50, 0.70, 0.85]

VARIANTS = [
    ("gaussian_independent", "gaussian", False),
    ("rademacher_independent", "rademacher", False),
    ("hadamard_independent", "hadamard", False),
    ("gaussian_paired_chunk", "gaussian", True),
    ("rademacher_paired_chunk", "rademacher", True),
    ("hadamard_paired_chunk", "hadamard", True),
]
SOURCE_CONTROL = "gaussian_independent"
SOURCE_SIGMA = 0.20


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_source(root: Path) -> dict[str, str]:
    head = _run("git", "rev-parse", "HEAD", cwd=root)
    if head != DUST_COMMIT:
        raise RuntimeError(f"DUST source commit mismatch: {head} != {DUST_COMMIT}")
    observed = {}
    for rel, expected in DUST_BLOBS.items():
        blob = _run("git", "hash-object", rel, cwd=root)
        if blob != expected:
            raise RuntimeError(f"DUST blob mismatch for {rel}: {blob} != {expected}")
        observed[rel] = blob
    return {"commit": head, **observed}


def load_dust(root: Path):
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("dust")
        if Path(module.__file__).resolve() != (root / "dust.py").resolve():
            raise RuntimeError(f"loaded unexpected dust module: {module.__file__}")
        return module
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)


def hadamard(dimension: int, *, device: torch.device) -> torch.Tensor:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < dimension:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix


class NoisePlanner:
    """Generate reproducible independent or in-chunk paired directions lazily."""

    def __init__(
        self,
        family: str,
        *,
        seeds: list[int],
        length: int,
        width: int,
        paired: bool,
        device: torch.device,
    ) -> None:
        self.family = family
        self.seeds = seeds
        self.batch = len(seeds)
        self.length = length
        self.width = width
        self.paired = paired
        self.device = device
        self.draws_emitted = 0
        self.unique_index = 0
        self.generators = [
            torch.Generator(device=device).manual_seed(seed) for seed in seeds
        ]
        self.matrix = None
        self.rows = None
        self.signs = None
        if family == "hadamard":
            self.matrix = hadamard(width, device=device)
            rows = []
            signs = []
            for generator in self.generators:
                rows.append(torch.randperm(width, generator=generator, device=device))
                sign_bits = torch.randint(
                    0,
                    2,
                    (length, width),
                    generator=generator,
                    device=device,
                )
                signs.append(sign_bits.mul_(2).sub_(1).float())
            self.rows = torch.stack(rows, dim=0)
            self.signs = torch.stack(signs, dim=0)

    def _base(self, batch_index: int, direction_index: int) -> torch.Tensor:
        generator = self.generators[batch_index]
        if self.family == "gaussian":
            return torch.randn(
                (self.length, self.width),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
        if self.family == "rademacher":
            bits = torch.randint(
                0,
                2,
                (self.length, self.width),
                generator=generator,
                device=self.device,
            )
            return bits.mul_(2).sub_(1).float()
        if self.family == "hadamard":
            assert self.matrix is not None and self.rows is not None and self.signs is not None
            row = int(self.rows[batch_index, direction_index % self.width].item())
            return self.matrix[row][None, :] * self.signs[batch_index]
        raise ValueError(f"unknown family: {self.family}")

    def next(self, nrep: int, dtype: torch.dtype) -> torch.Tensor:
        if self.paired:
            if nrep != 2:
                raise RuntimeError("paired-chunk registration requires source nrep=2")
            base = torch.stack(
                [
                    self._base(batch_index, self.unique_index)
                    for batch_index in range(self.batch)
                ],
                dim=0,
            )
            self.unique_index += 1
            result = torch.stack((base, -base), dim=0)
        else:
            draws = []
            for local_draw in range(nrep):
                direction_index = self.draws_emitted + local_draw
                draws.append(
                    torch.stack(
                        [
                            self._base(batch_index, direction_index)
                            for batch_index in range(self.batch)
                        ],
                        dim=0,
                    )
                )
            result = torch.stack(draws, dim=0)
        self.draws_emitted += nrep
        return result.reshape(nrep * self.batch, self.length, self.width).to(dtype=dtype)


@contextmanager
def planned_noise(engine, planner: NoisePlanner, expected_draws: int) -> Iterator[None]:
    original = engine.noise

    def noise(*shape: int, dtype=None):
        nrep = shape[0] // engine.batch
        if shape[1:] != (planner.length, planner.width):
            raise RuntimeError(f"unexpected noise shape: {shape}")
        return planner.next(nrep, dtype or engine.dtype)

    engine.noise = noise
    try:
        yield
    finally:
        engine.noise = original
    if planner.draws_emitted != expected_draws:
        raise RuntimeError(
            f"noise-plan draw mismatch: {planner.draws_emitted} != {expected_draws}"
        )


def source_estimate(
    engine,
    *,
    site: str,
    family: str,
    paired: bool,
    seeds: list[int],
    draws: int,
    sigma: float,
) -> torch.Tensor:
    if draws % engine.settings.nrep:
        raise ValueError("draw count must be divisible by source nrep")
    if paired and draws % 2:
        raise ValueError("paired draw count must be even")
    width = engine.outputs[site].shape[-1]
    planner = NoisePlanner(
        family,
        seeds=seeds,
        length=engine.length,
        width=width,
        paired=paired,
        device=engine.device,
    )
    with planned_noise(engine, planner, draws):
        return engine.direct_errors([site], draws, sigma)[site].float()


def reference_gradients(
    engine,
    *,
    site: str,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same-token Jacobian diagonal and total-loss gradient for batch element zero."""
    module = engine.modules[site]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        leaf = output.detach().requires_grad_(True)
        captured["activation"] = leaf
        return leaf

    handle = module.register_forward_hook(hook)
    try:
        losses = engine.losses(engine.x, engine.y)
        activation = captured["activation"]
        direct_rows = []
        for position in positions:
            grad = torch.autograd.grad(
                losses[0, position],
                activation,
                retain_graph=True,
                create_graph=False,
            )[0]
            direct_rows.append(grad[0, position].float())
        valid = engine.y != -1
        total = torch.autograd.grad(losses[valid].sum(), activation)[0].float()
    finally:
        handle.remove()
    return torch.stack(direct_rows, dim=0), total[0, positions, :]


def metric_values(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    estimate = estimate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    denom = estimate.norm() * reference.norm()
    cosine = torch.dot(estimate, reference) / denom.clamp_min(1e-20)
    relative = (estimate - reference).norm() / reference.norm().clamp_min(1e-20)
    return {
        "cosine": float(cosine.item()),
        "relative_l2": float(relative.item()),
        "estimate_norm": float(estimate.norm().item()),
        "reference_norm": float(reference.norm().item()),
    }


def make_tokens(
    config,
    *,
    batch_seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(batch_seed)
    one = torch.randint(
        0,
        config.vocab_size,
        (1, SEQUENCE_LENGTH + 1),
        generator=generator,
        dtype=torch.long,
    )
    x = one[:, :-1].repeat(batch_size, 1).contiguous().to(device)
    y = one[:, 1:].repeat(batch_size, 1).contiguous().to(device)
    return x, y


def sampled_positions() -> list[int]:
    return (
        torch.linspace(0, SEQUENCE_LENGTH - 1, steps=SAMPLED_TOKENS)
        .round()
        .long()
        .unique()
        .tolist()
    )


def site_class(site: str) -> str:
    return "mlp_hidden" if ".mlp.c_fc" in site else "writer_output_projection"


def execute_cells(
    engine,
    *,
    sites: list[str],
    estimator_seeds: list[int],
    batch_seed: int,
    k_values: list[int],
    sigma_by_variant: dict[str, list[float]],
    phase: str,
) -> list[dict]:
    x, y = make_tokens(
        engine.model.config,
        batch_seed=batch_seed,
        batch_size=len(estimator_seeds),
        device=engine.device,
    )
    engine.capture(x, y)
    positions = sampled_positions()
    rows: list[dict] = []

    for site in sites:
        direct_ref, total_ref = reference_gradients(
            engine,
            site=site,
            positions=positions,
        )
        width = int(engine.outputs[site].shape[-1])
        for variant, family, paired in VARIANTS:
            for sigma in sigma_by_variant[variant]:
                for draws in k_values:
                    if family == "hadamard":
                        unique = draws // 2 if paired else draws
                        if unique > width:
                            continue
                    if engine.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(engine.device)
                        torch.cuda.synchronize(engine.device)
                    started = time.perf_counter()
                    estimate = source_estimate(
                        engine,
                        site=site,
                        family=family,
                        paired=paired,
                        seeds=estimator_seeds,
                        draws=draws,
                        sigma=sigma,
                    )
                    if engine.device.type == "cuda":
                        torch.cuda.synchronize(engine.device)
                        peak_memory = int(torch.cuda.max_memory_allocated(engine.device))
                    else:
                        peak_memory = None
                    elapsed = time.perf_counter() - started
                    selected = estimate[:, positions, :]
                    for batch_index, estimator_seed in enumerate(estimator_seeds):
                        rows.append(
                            {
                                "phase": phase,
                                "batch_seed": batch_seed,
                                "site": site,
                                "site_class": site_class(site),
                                "activation_width": width,
                                "activation_shape": list(engine.outputs[site].shape),
                                "sampled_tokens": positions,
                                "variant": variant,
                                "family": family,
                                "paired_chunk": paired,
                                "sigma": sigma,
                                "population_draws": draws,
                                "unique_direction_count": draws // 2 if paired else draws,
                                "draw_equivalent_forward_evaluations": draws,
                                "physical_forward_calls": draws // engine.settings.nrep,
                                "source_nrep": engine.settings.nrep,
                                "estimator_seed": estimator_seed,
                                "cell_elapsed_seconds": elapsed,
                                "cell_peak_device_memory_bytes": peak_memory,
                                "direct_reference": metric_values(
                                    selected[batch_index],
                                    direct_ref,
                                ),
                                "total_loss_reference": metric_values(
                                    selected[batch_index],
                                    total_ref,
                                ),
                            }
                        )
    return rows


def select_sigmas(rows: list[dict]) -> tuple[dict[str, float], dict[str, dict]]:
    selected = {SOURCE_CONTROL: SOURCE_SIGMA}
    evidence: dict[str, dict] = {}
    for variant, _family, _paired in VARIANTS:
        if variant == SOURCE_CONTROL:
            evidence[variant] = {
                "selection": "source_locked",
                "sigma": SOURCE_SIGMA,
            }
            continue
        candidates = []
        for sigma in SIGMA_GRID:
            subset = [r for r in rows if r["variant"] == variant and r["sigma"] == sigma]
            if not subset:
                continue
            mean_cosine = statistics.fmean(
                r["direct_reference"]["cosine"] for r in subset
            )
            mean_relative = statistics.fmean(
                r["direct_reference"]["relative_l2"] for r in subset
            )
            candidates.append(
                {
                    "sigma": sigma,
                    "mean_direct_cosine": mean_cosine,
                    "mean_direct_relative_l2": mean_relative,
                    "rows": len(subset),
                }
            )
        if not candidates:
            raise RuntimeError(f"no calibration rows for {variant}")
        candidates.sort(
            key=lambda row: (
                -row["mean_direct_cosine"],
                row["mean_direct_relative_l2"],
                row["sigma"],
            )
        )
        selected[variant] = float(candidates[0]["sigma"])
        evidence[variant] = {
            "selection": "highest_mean_direct_cosine_then_relative_l2_then_smaller_sigma",
            "selected": candidates[0],
            "candidates": candidates,
        }
    return selected, evidence


def median_metric(
    rows: list[dict],
    *,
    site: str,
    variant: str,
    draws: int,
    metric: str,
) -> float:
    values = [
        r["direct_reference"][metric]
        for r in rows
        if r["site"] == site
        and r["variant"] == variant
        and r["population_draws"] == draws
    ]
    if not values:
        raise RuntimeError(f"missing rows for {site} {variant} K={draws}")
    return float(statistics.median(values))


def crossing(
    rows: list[dict],
    *,
    site: str,
    variant: str,
    target: float,
) -> dict | None:
    for draws in HELDOUT_K:
        cosine = median_metric(
            rows,
            site=site,
            variant=variant,
            draws=draws,
            metric="cosine",
        )
        if cosine >= target:
            return {
                "draws": draws,
                "median_cosine": cosine,
                "median_relative_l2": median_metric(
                    rows,
                    site=site,
                    variant=variant,
                    draws=draws,
                    metric="relative_l2",
                ),
            }
    return None


def analyze_heldout(rows: list[dict]) -> dict:
    ratios: list[dict] = []
    candidates = [variant for variant, _family, _paired in VARIANTS if variant != SOURCE_CONTROL]
    for site in HELDOUT_SITES:
        for target in TARGET_COSINES:
            baseline = crossing(
                rows,
                site=site,
                variant=SOURCE_CONTROL,
                target=target,
            )
            for variant in candidates:
                candidate = crossing(
                    rows,
                    site=site,
                    variant=variant,
                    target=target,
                )
                if baseline is None:
                    ratio = None
                    l2_ratio = None
                    status = "BASELINE_CENSORED"
                elif candidate is None:
                    ratio = 0.0
                    l2_ratio = None
                    status = "CANDIDATE_CENSORED"
                else:
                    ratio = baseline["draws"] / candidate["draws"]
                    l2_ratio = (
                        candidate["median_relative_l2"]
                        / max(baseline["median_relative_l2"], 1e-20)
                    )
                    status = "OBSERVED"
                ratios.append(
                    {
                        "site": site,
                        "target_cosine": target,
                        "variant": variant,
                        "status": status,
                        "baseline": baseline,
                        "candidate": candidate,
                        "efficiency_ratio": ratio,
                        "relative_l2_ratio": l2_ratio,
                    }
                )

    summaries = {}
    for variant in candidates:
        subset = [
            row
            for row in ratios
            if row["variant"] == variant and row["status"] != "BASELINE_CENSORED"
        ]
        ratio_values = [float(row["efficiency_ratio"]) for row in subset]
        observed_l2 = [
            float(row["relative_l2_ratio"])
            for row in subset
            if row["relative_l2_ratio"] is not None
        ]
        sites_at_2x = sorted(
            {
                row["site"]
                for row in subset
                if row["efficiency_ratio"] is not None
                and row["efficiency_ratio"] >= 2.0
            }
        )
        median_ratio = statistics.median(ratio_values) if ratio_values else None
        median_l2_ratio = statistics.median(observed_l2) if observed_l2 else None
        safeguard = median_l2_ratio is not None and median_l2_ratio <= 1.10
        advance = (
            median_ratio is not None
            and median_ratio >= 2.0
            and len(sites_at_2x) >= 2
            and safeguard
        )
        strong = (
            median_ratio is not None
            and median_ratio >= 4.0
            and sum(1 for value in ratio_values if value >= 4.0) > len(ratio_values) / 2
            and safeguard
        )
        summaries[variant] = {
            "baseline_reachable_strata": len(subset),
            "median_efficiency_ratio": median_ratio,
            "median_relative_l2_ratio_at_observed_crossings": median_l2_ratio,
            "relative_l2_safeguard_pass": safeguard,
            "sites_with_ratio_at_least_2": sites_at_2x,
            "advance": advance,
            "strong_advance": strong,
        }

    advancing = [variant for variant, row in summaries.items() if row["advance"]]
    return {
        "targets": TARGET_COSINES,
        "ratios": ratios,
        "candidate_summaries": summaries,
        "advancing_variants": advancing,
        "disposition": "WP01_A_ADVANCES_TO_WP01_B" if advancing else "STOPPED_AT_GATE1",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_record(
    *,
    source: dict[str, str],
    device: torch.device,
    settings,
    phase: str,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "experiment_id": f"PS-DUST-WP01-A-{phase.upper()}-001",
        "phase": phase,
        "claim_boundary": (
            "Initialization-only estimator screen. No transformer-training, scaling, "
            "novelty, deployment, commercial, or backprop-superiority claim."
        ),
        "protocol": {
            "tracker": "grandchallenge/MODULUS#26",
            "amendments": ["A001", "A002", "A003", "A004"],
            "primary_reference": "same_token_local_autodiff",
            "targets": TARGET_COSINES,
        },
        "source_lock": source,
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        "model": {
            "model_seed": MODEL_SEED,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "source_semantics": {
            "nrep": settings.nrep,
            "draw_equivalent_ledger": (
                "K perturbation members per row; physical perturbation model calls = K/nrep."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = verify_source(args.dust_root.resolve())
    dust = load_dust(args.dust_root.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config = dust.GPTConfig(sequence_len=SEQUENCE_LENGTH)
    model = dust.make_model(config, device, MODEL_SEED)
    model.eval()
    settings = dust.recipe("1M", 256)
    if settings.nrep != 2:
        raise RuntimeError(f"A003 requires source nrep=2, observed {settings.nrep}")
    engine = dust.DUST(model, settings, seed=MODEL_SEED)

    record = common_record(source=source, device=device, settings=settings, phase=args.phase)
    started = time.perf_counter()
    try:
        if args.phase == "calibration":
            sigma_by_variant = {variant: list(SIGMA_GRID) for variant, _f, _p in VARIANTS}
            rows = execute_cells(
                engine,
                sites=CALIBRATION_SITES,
                estimator_seeds=CALIBRATION_ESTIMATOR_SEEDS,
                batch_seed=CALIBRATION_BATCH_SEED,
                k_values=CALIBRATION_K,
                sigma_by_variant=sigma_by_variant,
                phase="calibration",
            )
            selected, evidence = select_sigmas(rows)
            record.update(
                {
                    "batch_seed": CALIBRATION_BATCH_SEED,
                    "sites": CALIBRATION_SITES,
                    "estimator_seeds": CALIBRATION_ESTIMATOR_SEEDS,
                    "k_values": CALIBRATION_K,
                    "sigma_grid": SIGMA_GRID,
                    "selected_sigmas": selected,
                    "selection_evidence": evidence,
                    "rows": rows,
                }
            )
        else:
            if args.calibration_artifact is None:
                raise ValueError("--calibration-artifact is required for heldout phase")
            calibration = json.loads(args.calibration_artifact.read_text(encoding="utf-8"))
            if calibration.get("phase") != "calibration":
                raise RuntimeError("calibration artifact phase mismatch")
            if calibration.get("source_lock", {}).get("commit") != DUST_COMMIT:
                raise RuntimeError("calibration source-lock mismatch")
            selected = calibration.get("selected_sigmas")
            expected_variants = {variant for variant, _f, _p in VARIANTS}
            if set(selected or {}) != expected_variants:
                raise RuntimeError("calibration sigma map is incomplete")
            sigma_by_variant = {
                variant: [float(selected[variant])] for variant in expected_variants
            }
            rows = execute_cells(
                engine,
                sites=HELDOUT_SITES,
                estimator_seeds=HELDOUT_ESTIMATOR_SEEDS,
                batch_seed=HELDOUT_BATCH_SEED,
                k_values=HELDOUT_K,
                sigma_by_variant=sigma_by_variant,
                phase="heldout",
            )
            decision = analyze_heldout(rows)
            record.update(
                {
                    "batch_seed": HELDOUT_BATCH_SEED,
                    "sites": HELDOUT_SITES,
                    "estimator_seeds": HELDOUT_ESTIMATOR_SEEDS,
                    "k_values": HELDOUT_K,
                    "frozen_sigmas": selected,
                    "calibration_artifact_sha256": sha256_file(args.calibration_artifact),
                    "rows": rows,
                    "decision": decision,
                }
            )
    finally:
        engine.close()

    record["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(args.output)
    print(
        json.dumps(
            {
                "experiment_id": record["experiment_id"],
                "phase": args.phase,
                "rows": len(record["rows"]),
                "elapsed_seconds": record["elapsed_seconds"],
                "output": str(args.output),
                "sha256": digest,
                "disposition": record.get("decision", {}).get("disposition"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
