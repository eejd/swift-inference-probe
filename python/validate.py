#!/usr/bin/env python3
"""NaN/Inf op scan — and a check on whether that scan can detect anything.

`MLModelValidator.find_failing_ops_with_nan_output()` and
`_infinite_output()` are the detectors behind TEST_STRATEGY 5.1's Tier 1 rule:
"NaN / Inf in output. Unconditional failure, no tolerance debate."

Running them on a nominal input reports zero failing ops, which looks like a
pass. It is not a pass until you know the detector *can* fire — a scan that
never fires and a model that never breaks are indistinguishable from the
report alone. That is the manipulation-check discipline from TESTING_PRIMER
4.5, applied to a detector instead of an experiment.

So this runs three things together, and the third is what makes the first
interpretable:

  1. SCAN      — the NaN and Inf op scans over a battery of inputs.
  2. BEHAVIOUR — what the model actually returned for each input: finite?
                 top-1? output magnitude? A detector reporting "clean" while
                 the output is garbage is the case worth catching.
  3. CONTROL   — deliberately malformed inputs (all-NaN, all-Inf) plus a
                 PyTorch reference on the same input. PyTorch propagates NaN
                 per IEEE; if Core ML does not, the detector has no coverage
                 and the finding is about the detector, not the model.

Usage:
    python validate.py                          # fp32 + fp16, .cpuOnly
    python validate.py --units cpuOnly cpuAndNE
    python validate.py --no-torch-reference
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

import coremltools as ct
from coremltools.models.ml_program.experimental.debugging_utils import MLModelValidator

# Same reasoning as localize.py: the scan drives a tqdm bar per input.
if not sys.stderr.isatty():
    tqdm.tqdm.__init__ = functools.partialmethod(tqdm.tqdm.__init__, disable=True)

# Repository-relative, not cwd-relative — see export_model.py and F-011.
FIXTURES = Path(__file__).resolve().parent.parent / "Fixtures" / "models"

SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

UNITS = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
    "cpuAndNE": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}

# fp16's largest finite value is 65504, so a network whose activations are
# driven past that has to do *something*. The sweep exists to find out what,
# and to locate the input magnitude where it starts.
SCALES = [1.0, 1e1, 1e2, 1e3, 1e4, 1e6, 1e15, 1e30, 1e37]


def base_input() -> np.ndarray:
    """The same seeded, ImageNet-scaled input the other experiments use."""
    torch.manual_seed(SEED)
    return ((torch.rand(*INPUT_SHAPE) - IMAGENET_MEAN) / IMAGENET_STD).numpy()


def build_battery(base: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """(name, kind, tensor). `kind` separates controls from ordinary inputs."""
    battery: list[tuple[str, str, np.ndarray]] = [
        ("zeros", "edge", np.zeros_like(base)),
    ]
    for s in SCALES:
        label = "nominal" if s == 1.0 else f"scale_{s:g}"
        battery.append((label, "scaled", (base * s).astype(np.float32)))

    # Controls. If the detector cannot fire on these, it cannot fire at all.
    mixed_nan = base.copy().ravel(); mixed_nan[:100] = np.nan
    mixed_inf = base.copy().ravel(); mixed_inf[:100] = np.inf
    battery += [
        ("all_nan", "control", np.full_like(base, np.nan)),
        ("all_inf", "control", np.full_like(base, np.inf)),
        ("mixed_nan", "control", mixed_nan.reshape(base.shape).astype(np.float32)),
        ("mixed_inf", "control", mixed_inf.reshape(base.shape).astype(np.float32)),
    ]
    return battery


def op_label(op) -> str:
    name = next((o.name for o in op.outputs), "?")
    return f"{op.type}:{name}"


def describe_output(y: np.ndarray) -> dict:
    finite = np.isfinite(y)
    return {
        "all_finite": bool(finite.all()),
        "nan_count": int(np.isnan(y).sum()),
        "inf_count": int(np.isinf(y).sum()),
        "top1": int(y.argmax()) if finite.any() else -1,
        "abs_max": float(np.abs(y[finite]).max()) if finite.any() else float("nan"),
    }


async def scan(validator, x: np.ndarray) -> dict:
    nan_ops = await validator.find_failing_ops_with_nan_output(inputs={INPUT_NAME: x})
    inf_ops = await validator.find_failing_ops_with_infinite_output(inputs={INPUT_NAME: x})
    return {
        "nan_op_count": len(nan_ops),
        "inf_op_count": len(inf_ops),
        "nan_ops": sorted(op_label(o) for o in nan_ops),
        "inf_ops": sorted(op_label(o) for o in inf_ops),
    }


def torch_reference(base: np.ndarray, battery) -> dict:
    """PyTorch fp32 on the same inputs — the standard for correct IEEE behaviour.

    This is the reference that turns "Core ML reported no NaN" into a finding.
    If PyTorch propagates NaN on an input where Core ML returns a confident
    finite answer, the two runtimes disagree about what malformed input means.
    """
    import torchvision
    model = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).eval()
    out = {}
    for name, _kind, x in battery:
        with torch.no_grad():
            y = model(torch.from_numpy(x)).numpy().ravel()
        out[name] = describe_output(y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--units", nargs="+", default=["cpuOnly"], choices=list(UNITS))
    ap.add_argument("--no-torch-reference", action="store_true")
    ap.add_argument("--models-dir", default=str(FIXTURES))
    ap.add_argument("--out", default=str(FIXTURES / "nan_inf_validation.json"))
    args = ap.parse_args()

    md = Path(args.models_dir)
    base = base_input()
    battery = build_battery(base)
    report: dict = {"seed": SEED, "scales": SCALES,
                    "battery": [n for n, _k, _x in battery], "runs": {}}

    detector_ever_fired = False

    for precision in args.precisions:
        path = str(md / f"mobilenet_v3_large.{precision}.mlpackage")
        for unit in args.units:
            key = f"{precision}.{unit}"
            model = ct.models.MLModel(path, compute_units=UNITS[unit])
            validator = MLModelValidator(model)
            print(f"\n=== {key} ===")
            print(f"  {'input':<14s} {'finite':>6s} {'nan':>4s} {'inf':>4s} "
                  f"{'top1':>5s} {'|max|':>11s}   scan(nan/inf ops)")
            rows = {}
            for name, kind, x in battery:
                y = np.asarray(next(iter(model.predict({INPUT_NAME: x}).values()))).ravel()
                behaviour = describe_output(y)
                found = asyncio.run(scan(validator, x))
                if found["nan_op_count"] or found["inf_op_count"]:
                    detector_ever_fired = True
                rows[name] = {"kind": kind, "behaviour": behaviour, "scan": found}
                print(f"  {name:<14s} {str(behaviour['all_finite']):>6s} "
                      f"{behaviour['nan_count']:>4d} {behaviour['inf_count']:>4d} "
                      f"{behaviour['top1']:>5d} {behaviour['abs_max']:>11.4g}   "
                      f"{found['nan_op_count']}/{found['inf_op_count']}")
            report["runs"][key] = rows

    if not args.no_torch_reference:
        print("\n=== pytorch fp32 reference (same inputs) ===")
        ref = torch_reference(base, battery)
        report["torch_fp32_reference"] = ref
        for name, b in ref.items():
            print(f"  {name:<14s} finite={str(b['all_finite']):<5s} "
                  f"nan={b['nan_count']:<5d} inf={b['inf_count']:<5d} top1={b['top1']}")

    report["detector_ever_fired"] = detector_ever_fired
    print("\n" + "=" * 60)
    if detector_ever_fired:
        print("NaN/Inf detector FIRED at least once — it has coverage here.")
    else:
        print("NaN/Inf detector NEVER FIRED, including on all-NaN and all-Inf")
        print("input. On this configuration it provides no coverage, and a")
        print("clean report from it is not evidence of a healthy model.")

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
