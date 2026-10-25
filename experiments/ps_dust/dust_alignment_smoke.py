#!/usr/bin/env python3
"""Source-locked PS-DUST Gate-1 engineering smoke.

This runner intentionally does not make a Gate-1 scientific decision. It verifies
that the source DUST Gaussian estimator, structured perturbation variants, and two
reference-gradient semantics can be measured on the same frozen model/site/batch.

The scientific held-out matrix is governed by docs/research/PS_DUST_WP01_PROTOCOL.md
and amendment A001. Smoke rows MUST NOT be pooled into the held-out decision set.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Iterator

import torch

DUST_COMMIT = "2fdb01ba91ca38368a2b6b31a2697d77512ad175"
DUST_BLOBS = {
    "dust.py": "c7d08ee9edc3a51dfbd9e73bbf787b9ca819c5c2",
    "model.py": "e605727f6ae9bb114647ae04308d98d0ba5b9dee",
}


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


def directions(
    family: str,
    *,
    population: int,
    batch: int,
    length: int,
    width: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (population, batch, length, width)
    if family == "gaussian":
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if family == "rademacher":
        bits = torch.randint(0, 2, shape, device=device, generator=generator)
        return bits.mul_(2).sub_(1).float()
    if family == "hadamard":
        if population > width:
            raise ValueError("Hadamard population cannot exceed activation width")
        matrix = hadamard(width, device=device)
        rows = torch.randperm(width, generator=generator, device=device)[:population]
        signs = torch.randint(
            0,
            2,
            (batch, length, width),
            device=device,
            generator=generator,
        ).mul_(2).sub_(1).float()
        return matrix[rows, None, None, :] * signs[None, :, :, :]
    raise ValueError(f"unknown direction family: {family}")


@contextmanager
def planned_noise(engine, plan: torch.Tensor) -> Iterator[None]:
    """Temporarily feed a precomputed direction plan through source DUST.noise."""
    original = engine.noise
    offset = 0

    def noise(*shape: int, dtype=None):
        nonlocal offset
        count = shape[0] // engine.batch
        chunk = plan[offset : offset + count]
        if chunk.shape[0] != count:
            raise RuntimeError("structured noise plan exhausted")
        offset += count
        result = chunk.reshape(shape)
        return result.to(dtype=dtype or engine.dtype)

    engine.noise = noise
    try:
        yield
    finally:
        engine.noise = original
    if offset != plan.shape[0]:
        raise RuntimeError(f"structured noise plan partially consumed: {offset}/{plan.shape[0]}")


def source_one_sided(engine, site: str, plan: torch.Tensor, sigma: float) -> torch.Tensor:
    with planned_noise(engine, plan):
        return engine.direct_errors([site], plan.shape[0], sigma)[site].float()


def antithetic(engine, site: str, plan: torch.Tensor, sigma: float) -> torch.Tensor:
    n = engine.settings.nrep
    if plan.shape[0] % n:
        raise ValueError("population must be divisible by source nrep")
    x = engine.x.repeat(n, 1)
    y = engine.y.repeat(n, 1)
    error = torch.zeros(
        engine.batch,
        engine.length,
        plan.shape[-1],
        device=engine.device,
        dtype=torch.float32,
    )
    for start in range(0, plan.shape[0], n):
        a = plan[start : start + n].reshape(n * engine.batch, engine.length, -1)
        losses = []
        for sign in (1.0, -1.0):
            engine.jitter = {site: sign * sigma * a}
            try:
                losses.append(engine.losses(x, y).view(n, engine.batch, engine.length))
            finally:
                engine.jitter = {}
        directional = (losses[0] - losses[1]) / (2.0 * sigma)
        error += torch.einsum(
            "nbt,nbtd->btd",
            directional.to(a.dtype),
            a.view(n, engine.batch, engine.length, -1),
        ).float()
    return error / plan.shape[0]


def reference_gradients(
    engine,
    site: str,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return same-token Jacobian diagonal and total-loss gradient on selected tokens."""
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
    direct = torch.stack(direct_rows, dim=0)
    total_selected = total[0, positions, :]
    return direct, total_selected


def metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--site", default="transformer.h.0.attn.c_proj")
    parser.add_argument("--populations", default="16,32,64")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--batch-seed", type=int, default=314159)
    parser.add_argument("--sampled-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = verify_source(args.dust_root.resolve())
    dust = load_dust(args.dust_root.resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config = dust.GPTConfig(sequence_len=args.sequence_length)
    model = dust.make_model(config, device, args.model_seed)
    settings = dust.recipe("1M", 256)
    engine = dust.DUST(model, settings, seed=args.model_seed)

    cpu_generator = torch.Generator(device="cpu").manual_seed(args.batch_seed)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, args.sequence_length + 1),
        generator=cpu_generator,
        dtype=torch.long,
    )
    x = tokens[:, :-1].contiguous().to(device)
    y = tokens[:, 1:].contiguous().to(device)
    engine.capture(x, y)

    sampled = torch.linspace(
        0,
        args.sequence_length - 1,
        steps=min(args.sampled_tokens, args.sequence_length),
    ).round().long().unique().tolist()
    direct_ref, total_ref = reference_gradients(engine, args.site, sampled)
    populations = [int(value) for value in args.populations.split(",") if value]
    rows = []
    width = engine.outputs[args.site].shape[-1]

    start_time = time.perf_counter()
    for seed_index in range(args.seeds):
        seed = 1000 + seed_index
        for family in ("gaussian", "rademacher", "hadamard"):
            for population in populations:
                plan = directions(
                    family,
                    population=population,
                    batch=engine.batch,
                    length=engine.length,
                    width=width,
                    seed=seed,
                    device=device,
                )
                one = source_one_sided(engine, args.site, plan, args.sigma)
                selected = one[0, sampled, :]
                row = {
                    "family": family,
                    "antithetic": False,
                    "seed": seed,
                    "population": population,
                    "sigma": args.sigma,
                    "perturbed_forward_evaluations": population,
                    "forward_evaluations_including_clean": population + 1,
                    "direct_reference": metrics(selected, direct_ref),
                    "total_loss_reference": metrics(selected, total_ref),
                }
                rows.append(row)

                paired = antithetic(engine, args.site, plan, args.sigma)
                paired_selected = paired[0, sampled, :]
                rows.append(
                    {
                        "family": family,
                        "antithetic": True,
                        "seed": seed,
                        "population": population,
                        "sigma": args.sigma,
                        "perturbed_forward_evaluations": 2 * population,
                        "forward_evaluations_including_clean": 2 * population + 1,
                        "direct_reference": metrics(paired_selected, direct_ref),
                        "total_loss_reference": metrics(paired_selected, total_ref),
                    }
                )

    elapsed = time.perf_counter() - start_time
    result = {
        "schema_version": "1.0.0",
        "experiment_id": "PS-DUST-WP01-ENGINEERING-SMOKE-001",
        "status": "ENGINEERING_SMOKE_ONLY",
        "claim_boundary": (
            "Initialization/random-token smoke only. These rows are excluded from Gate-1 "
            "calibration/held-out decisions and cannot satisfy an advance criterion."
        ),
        "source_lock": source,
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "elapsed_seconds": elapsed,
        },
        "model": {
            "sequence_length": args.sequence_length,
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "site": args.site,
            "model_seed": args.model_seed,
            "batch_seed": args.batch_seed,
            "sampled_tokens": sampled,
        },
        "source_semantics": {
            "nrep": settings.nrep,
            "note": (
                "One-sided rows use source DUST.direct_errors with a supplied noise plan, "
                "including source chunk-centering semantics. Antithetic rows are the "
                "registered symmetric extension."
            ),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = _sha256(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"artifact_sha256={artifact_sha256}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
