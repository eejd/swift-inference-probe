#!/usr/bin/env python3
"""Localize numerical divergence to specific ops, swept across `atol`.

`MLModelComparator` answers "which op is responsible for the divergence?" — but
the answer is a *function of the tolerance you pass*. The #2625 reporter saw it
blame ordinary `linear`/`conv` layers at some tolerances and not others, which
means a single localization result is not a finding; it is a finding plus the
`atol` that produced it (docs/TEST_STRATEGY.md 4.5).

So this sweeps `atol` across decades and reports the flagged-op set at each.
Expected shape: everything flagged at very low `atol`, nothing at very high, and
a plateau in between where the answer stops changing. That plateau is the
defensible operating point, and the curve is the evidence for it.

Two arms, because the project measured the second effect to be the larger one:

  1. PRECISION  — Core ML fp32 (reference) vs fp16 (target), both on .cpuOnly.
                  Where does halving the mantissa first show up? This is the
                  #2625 question.
  2. COMPUTE    — the fp16 model on .cpuOnly (reference) vs .cpuAndNE (target).
                  Where do the CPU and Neural Engine first disagree? F-007/F-008
                  found this effect larger than precision drift, so localizing
                  it matters more, and it needs no second model file.

Usage:
    python localize.py                          # both arms, default sweep
    python localize.py --arm compute
    python localize.py --atols 1e-4 1e-3 1e-2
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

import coremltools as ct
from coremltools.models.ml_program.experimental.debugging_utils import MLModelComparator

# The comparator drives a tqdm bar over ~700 ops for every atol. That is useful
# interactively and is 89 KB of escape codes per sweep point in a redirected
# log, so keep it only when stderr is a terminal.
if not sys.stderr.isatty():
    tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)

# Repository-relative, not cwd-relative — see export_model.py and F-011.
FIXTURES = Path(__file__).resolve().parent.parent / "Fixtures" / "models"

SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Several decades, because the point is the shape of the curve rather than any
# single value. Sorted ascending so the report reads from strict to permissive.
DEFAULT_ATOLS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def make_input() -> np.ndarray:
    """The same seeded, ImageNet-scaled input the other experiments use."""
    torch.manual_seed(SEED)
    return ((torch.rand(*INPUT_SHAPE) - IMAGENET_MEAN) / IMAGENET_STD).numpy()


def op_label(op) -> str:
    """`type:output_name` — the type alone is not unique within a graph."""
    name = next((o.name for o in op.outputs), "?")
    return f"{op.type}:{name}"


async def sweep_arm(reference, target, x: np.ndarray, atols: list[float]) -> list[dict]:
    """Run find_failing_ops once per atol against one reference/target pair.

    `rtol` is pinned to 0 deliberately. np.allclose's tolerance is
    `atol + rtol * |target|`, so leaving rtol non-zero would make the effective
    threshold depend on activation magnitude and the sweep would no longer be
    varying one thing (docs/TEST_STRATEGY.md 4.3).
    """
    comparator = MLModelComparator(reference_model=reference, target_model=target)
    results = []

    for atol in atols:
        def compare_outputs(op, reference_output, target_output, _atol=atol) -> bool:
            # True means "these agree" — find_failing_ops collects the ops for
            # which this returns False and whose own inputs returned True.
            return bool(np.allclose(reference_output, target_output,
                                    rtol=0.0, atol=_atol))

        t0 = time.perf_counter()
        failing = await comparator.find_failing_ops(inputs={INPUT_NAME: x},
                                                    compare_outputs=compare_outputs)
        elapsed = time.perf_counter() - t0

        labels = sorted(op_label(op) for op in failing)
        types = sorted({op.type for op in failing})
        results.append({
            "atol": atol,
            "failing_op_count": len(labels),
            "failing_op_types": types,
            "failing_ops": labels,
            "seconds": round(elapsed, 2),
        })
        print(f"  atol={atol:<8g} {len(labels):3d} ops   "
              f"{', '.join(types) if types else '(none)':<40s} {elapsed:6.1f}s")

    return results


def find_plateaus(results: list[dict]) -> list[dict]:
    """Every run of >=2 consecutive atol values giving an identical op set.

    This is the operational definition of "localization stabilizes": not that
    the count is small, but that the *answer stops changing* as the parameter
    moves. A run of one is no plateau at all.

    All of them are reported, not just the widest. The measured data has two per
    arm — a strict-tolerance plateau where the search reaches the true origin,
    and a looser one where it reports a subset — and collapsing that to a single
    "the" plateau would hide the more useful fact that the two arms agree on the
    looser one. Which plateau is the operating point is a judgement, so the
    report should carry the evidence for both.
    """
    plateaus = []
    i = 0
    while i < len(results):
        j = i
        while j + 1 < len(results) and results[j + 1]["failing_ops"] == results[i]["failing_ops"]:
            j += 1
        if j > i:
            plateaus.append({
                "length": j - i + 1,
                "atol_low": results[i]["atol"],
                "atol_high": results[j]["atol"],
                "failing_op_count": results[i]["failing_op_count"],
                "failing_op_types": results[i]["failing_op_types"],
                "failing_ops": results[i]["failing_ops"],
            })
        i = j + 1
    return plateaus


def op_persistence(results: list[dict]) -> list[dict]:
    """How many sweep points flagged each op, most persistent first.

    A plateau says the answer is locally stable; this says which op survives the
    parameter changing at all. An op flagged across most of the sweep is the
    robust localization — it is the answer you would have got whatever tolerance
    you happened to pick.
    """
    counts: dict[str, int] = {}
    for r in results:
        for op in r["failing_ops"]:
            counts[op] = counts.get(op, 0) + 1
    return [{"op": op, "points": n, "of": len(results)}
            for op, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["precision", "compute", "both"], default="both")
    ap.add_argument("--atols", type=float, nargs="+", default=DEFAULT_ATOLS)
    ap.add_argument("--models-dir", default=str(FIXTURES))
    ap.add_argument("--out", default=str(FIXTURES / "op_localization.json"))
    args = ap.parse_args()

    md = Path(args.models_dir)
    fp32_path = str(md / "mobilenet_v3_large.fp32.mlpackage")
    fp16_path = str(md / "mobilenet_v3_large.fp16.mlpackage")
    atols = sorted(args.atols)
    x = make_input()

    report: dict = {"seed": SEED, "atols": atols, "rtol": 0.0, "arms": {}}

    arms = []
    if args.arm in ("precision", "both"):
        arms.append((
            "precision_fp32_vs_fp16",
            "Core ML fp32 vs fp16, both .cpuOnly — where fp16 conversion first shows",
            lambda: (ct.models.MLModel(fp32_path, compute_units=ct.ComputeUnit.CPU_ONLY),
                     ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_ONLY)),
        ))
    if args.arm in ("compute", "both"):
        arms.append((
            "compute_cpu_vs_ane",
            "fp16 .cpuOnly vs .cpuAndNE — where CPU and Neural Engine first disagree",
            lambda: (ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_ONLY),
                     ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)),
        ))

    for name, description, load in arms:
        print(f"\n=== {name} ===\n{description}")
        reference, target = load()
        results = asyncio.run(sweep_arm(reference, target, x, atols))
        plateaus = find_plateaus(results)
        persistence = op_persistence(results)
        report["arms"][name] = {"description": description,
                                "sweep": results,
                                "plateaus": plateaus,
                                "op_persistence": persistence}

        for p in plateaus:
            print(f"  → stable over atol {p['atol_low']:g}–{p['atol_high']:g} "
                  f"({p['length']} points): {p['failing_op_count']} ops "
                  f"{p['failing_op_types']}")
        if not plateaus:
            print("  → no plateau: the answer changes at every atol in this sweep")
        if persistence:
            top = persistence[0]
            print(f"  → most persistent: {top['op']} "
                  f"({top['points']}/{top['of']} sweep points)")

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
