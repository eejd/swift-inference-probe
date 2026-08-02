#!/usr/bin/env python3
"""Differential testing across Core ML compute units.

The same model, the same weights, the same input, four execution paths. They
must agree. No ground truth is needed — the question "which one is right?" is
replaced by "do they agree?", and disagreement is a defect regardless of which
path is wrong. This is what makes differential testing the answer to the oracle
problem (docs/TEST_STRATEGY.md 2.1).

Two measurements, and the second is what makes the first interpretable:

  1. NOISE FLOOR — the same input predicted repeatedly on ONE compute unit.
     Any spread here is the runtime disagreeing with itself.
  2. DIFFERENTIAL — many inputs, each unit compared against the .cpuOnly anchor.

A cross-unit difference only means something if it is larger than the noise
floor. Reporting (2) without (1) is how you end up asserting on jitter.

Usage:
    python compute_unit_differential.py                 # n=200, repeats=20
    python compute_unit_differential.py --n 50 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import coremltools as ct

SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# .cpuOnly first: it is the deterministic anchor every other unit is compared
# against (docs/SPEC.md 2.3).
UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
    "cpuAndNE": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}
ANCHOR = "cpuOnly"


def make_inputs(n: int) -> list[np.ndarray]:
    """Deterministic ImageNet-scaled inputs. Same seed => same tensors, always."""
    torch.manual_seed(SEED)
    return [
        ((torch.rand(*INPUT_SHAPE) - IMAGENET_MEAN) / IMAGENET_STD).numpy()
        for _ in range(n)
    ]


def predict(model, x: np.ndarray) -> np.ndarray:
    out = model.predict({INPUT_NAME: x})
    return np.asarray(next(iter(out.values()))).astype(np.float64).ravel()


def summarise(values: list[float]) -> dict:
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="../Fixtures/models/mobilenet_v3_large.fp16.mlpackage")
    ap.add_argument("--n", type=int, default=200, help="inputs for the differential")
    ap.add_argument("--repeats", type=int, default=20, help="repeats for the noise floor")
    ap.add_argument("--out", default="../Fixtures/models/compute_unit_differential.json")
    args = ap.parse_args()

    print(f"loading {Path(args.model).name} on {len(UNITS)} compute units ...")
    models = {}
    for name, unit in UNITS.items():
        t0 = time.perf_counter()
        models[name] = ct.models.MLModel(args.model, compute_units=unit)
        print(f"  {name:10s} loaded in {time.perf_counter()-t0:6.2f}s")

    report: dict = {"n_inputs": args.n, "repeats": args.repeats, "anchor": ANCHOR}

    # ---- 1. Noise floor -------------------------------------------------
    # Same input, same unit, many times. Anything non-zero here is the runtime
    # failing to agree with itself, and it lower-bounds every threshold we
    # could set (docs/TEST_STRATEGY.md 4.3).
    print(f"\n[1/2] noise floor — one input x {args.repeats} repeats per unit")
    fixed = make_inputs(1)[0]
    noise = {}
    for name, model in models.items():
        runs = [predict(model, fixed) for _ in range(args.repeats)]
        base = runs[0]
        spread = [float(np.abs(base - r).max()) for r in runs[1:]]
        worst = max(spread) if spread else 0.0
        noise[name] = {"max_abs_spread": worst,
                       "deterministic": bool(worst == 0.0)}
        verdict = "deterministic" if worst == 0.0 else f"varies by {worst:.3e}"
        print(f"  {name:10s} {verdict}")
    report["noise_floor"] = noise

    # ---- 2. Differential ------------------------------------------------
    print(f"\n[2/2] differential — {args.n} inputs, each unit vs .{ANCHOR}")
    inputs = make_inputs(args.n)

    outputs: dict[str, list[np.ndarray]] = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        outputs[name] = [predict(model, x) for x in inputs]
        print(f"  {name:10s} {args.n} predictions in {time.perf_counter()-t0:6.2f}s")

    anchor = outputs[ANCHOR]
    diffs: dict = {}
    for name, outs in outputs.items():
        if name == ANCHOR:
            continue
        l1, cos, top5, flips = [], [], [], 0
        for a, b in zip(anchor, outs):
            l1.append(float(np.abs(a - b).mean()))
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
            top5.append(len(set(np.argsort(a)[-5:]) & set(np.argsort(b)[-5:])) / 5.0)
            flips += int(a.argmax() != b.argmax())
        diffs[name] = {
            "l1_mean": summarise(l1),
            "cosine_min": float(min(cos)),
            "top5_overlap_mean": float(np.mean(top5)),
            "top5_overlap_min": float(min(top5)),
            "top1_flips": flips,
            "top1_flip_rate": flips / args.n,
        }
    report["differential"] = diffs

    # ---- Report ---------------------------------------------------------
    print(f"\n{'unit':12s} {'L1 mean':>10s} {'L1 p95':>10s} {'L1 max':>10s} "
          f"{'cos min':>10s} {'top1 flips':>11s}")
    print("-" * 68)
    for name, d in diffs.items():
        print(f"{name:12s} {d['l1_mean']['mean']:10.6f} {d['l1_mean']['p95']:10.6f} "
              f"{d['l1_mean']['max']:10.6f} {d['cosine_min']:10.6f} "
              f"{d['top1_flips']:6d}/{args.n:<4d}")

    # Signal vs noise. A disagreement smaller than the runtime's own jitter is
    # not evidence of anything.
    print("\nsignal vs noise:")
    for name, d in diffs.items():
        floor = max(noise[name]["max_abs_spread"], noise[ANCHOR]["max_abs_spread"])
        if floor == 0.0:
            print(f"  {name:12s} noise floor is exactly 0 — "
                  f"disagreement of {d['l1_mean']['mean']:.6f} is entirely systematic")
        else:
            print(f"  {name:12s} {d['l1_mean']['mean']/floor:8.1f}x the noise floor")

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
