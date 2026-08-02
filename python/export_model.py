#!/usr/bin/env python3
"""Export a torchvision model to Core ML at fp32 and fp16, and report drift.

Both precisions are exported deliberately. coremltools targets fp16 by default,
so divergence from the PyTorch fp32 reference is *expected* — the fp32 export is
what makes the fp16 delta attributable to precision rather than to conversion
(docs/SPEC.md 2.4).

Every artifact is written with a manifest recording resolved package versions and
the host chip. A drift number without that context is not interpretable
(docs/TEST_STRATEGY.md 4.4).

Usage:
    python export_model.py                     # MobileNetV3-Large
    python export_model.py --model resnet18 --output-dir ../Fixtures/models
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision

import coremltools as ct

# Fixed so the reference input is byte-identical across runs and machines.
# Every drift number below is relative to this exact tensor.
SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"

# Random pixels are normalised the way a real ImageNet input would be. Raw
# uniform [0,1] noise is off-distribution and produces logit magnitudes — and so
# L1 values — that are not comparable to figures measured on real images.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Manifest:
    """Provenance for an exported artifact. Mirrors HostEnvironment in Swift."""

    model: str
    weights: str
    precision: str
    input_shape: list[int]
    seed: int
    chip: str
    macos: str
    python: str
    torch: str
    torchvision: str
    coremltools: str
    numpy: str
    exported_at: str


def host_chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_model(name: str):
    """Load a pretrained torchvision classifier in eval mode."""
    builder = getattr(torchvision.models, name, None)
    if builder is None:
        sys.exit(f"unknown torchvision model: {name}")
    weights_enum = torchvision.models.get_model_weights(name).DEFAULT
    model = builder(weights=weights_enum).eval()
    return model, str(weights_enum)


def convert(traced, precision: str, sample: torch.Tensor):
    """Convert a traced model to a Core ML mlprogram at the given precision."""
    ct_precision = {
        "fp32": ct.precision.FLOAT32,
        "fp16": ct.precision.FLOAT16,
    }[precision]

    # TensorType, not ImageType: this isolates the runtime's numerics. Core ML's
    # built-in image preprocessing is a real thing to test, but it is a *separate*
    # test — folding it in here would confound precision drift with resize/scale
    # differences against the torchvision transform path.
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=sample.shape)],
        compute_precision=ct_precision,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        # Required, not an optimisation. ct.convert() implicitly loads and
        # compiles the result, and on coremltools 9.0 that implicit load
        # SEGFAULTS (exit 139) if any predict() has already run in the same
        # process — see docs/FINDINGS.md F-005. Skipping it also means the only
        # load we perform is the explicit, compute-unit-pinned one below, which
        # is what SPEC 2.3 wants anyway: the default load uses ComputeUnit.ALL,
        # which is not the deterministic reference path.
        skip_model_load=True,
    )


def drift(reference: np.ndarray, actual: np.ndarray) -> dict:
    """Comparison statistics matching TolerancePolicy.Criterion in Swift.

    Deliberately several statistics rather than one: each is blind to a failure
    mode the others catch (docs/TEST_STRATEGY.md 4.1).
    """
    ref = reference.astype(np.float64).ravel()
    act = actual.astype(np.float64).ravel()
    diff = np.abs(ref - act)

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    p, q = softmax(ref), softmax(act)
    nonzero = p > 0
    kl = float(np.sum(p[nonzero] * np.log(p[nonzero] / np.clip(q[nonzero], 1e-12, None))))

    ref_top5 = set(np.argsort(ref)[-5:].tolist())
    act_top5 = set(np.argsort(act)[-5:].tolist())

    return {
        "l1_mean": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine_similarity": float(ref @ act / (np.linalg.norm(ref) * np.linalg.norm(act))),
        "top1_reference": int(ref.argmax()),
        "top1_actual": int(act.argmax()),
        "top1_agrees": bool(ref.argmax() == act.argmax()),
        "top5_overlap": len(ref_top5 & act_top5) / 5.0,
        "kl_divergence_nats": kl,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mobilenet_v3_large")
    ap.add_argument("--output-dir", default="../Fixtures/models")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    sample = (torch.rand(*INPUT_SHAPE) - IMAGENET_MEAN) / IMAGENET_STD

    print(f"loading {args.model} ...")
    model, weights = load_model(args.model)

    with torch.no_grad():
        reference = model(sample).numpy()
    print(f"pytorch fp32 reference: shape={reference.shape} top1={reference.argmax()}")

    # The PyTorch fp32->fp16 baseline. This is the comparison that makes the
    # Core ML number interpretable: some drift is just what fp16 costs, and only
    # the *excess* over this baseline is attributable to the conversion.
    # Issue #2625 reports Core ML 0.0394 against this baseline's 0.0165 (~2.4x).
    torch_fp16_stats = None
    try:
        with torch.no_grad():
            half = copy.deepcopy(model).half()
            torch_fp16 = half(sample.half()).float().numpy()
        torch_fp16_stats = drift(reference, torch_fp16)
        print(f"pytorch fp16 baseline:  l1_mean={torch_fp16_stats['l1_mean']:.6f}")
    except Exception as exc:  # fp16 CPU support is uneven across torch versions
        print(f"pytorch fp16 baseline unavailable: {exc}")

    traced = torch.jit.trace(model, sample)

    results: dict[str, dict] = {}
    for precision in ("fp32", "fp16"):
        print(f"\nconverting {precision} ...")
        mlmodel = convert(traced, precision, sample)

        path = out / f"{args.model}.{precision}.mlpackage"
        mlmodel.save(str(path))

        manifest = Manifest(
            model=args.model,
            weights=weights,
            precision=precision,
            input_shape=list(INPUT_SHAPE),
            seed=SEED,
            chip=host_chip(),
            macos=platform.mac_ver()[0],
            python=sys.version.split()[0],
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            coremltools=ct.__version__,
            numpy=np.__version__,
            exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        (out / f"{args.model}.{precision}.manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2) + "\n"
        )

        # .cpuOnly is the deterministic reference path (docs/SPEC.md 2.3).
        # Reloading with an explicit compute unit rather than trusting the
        # default keeps this prediction reproducible.
        cpu_model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
        out_dict = cpu_model.predict({INPUT_NAME: sample.numpy()})
        actual = np.asarray(next(iter(out_dict.values())))

        stats = drift(reference, actual)
        results[precision] = stats
        print(f"  saved {path.name}")
        print(f"  l1_mean={stats['l1_mean']:.6f}  max_abs={stats['max_abs']:.6f}  "
              f"cos={stats['cosine_similarity']:.6f}  top1_agrees={stats['top1_agrees']}")

    # The comparison that matters. Dividing Core ML fp16 drift by Core ML *fp32*
    # drift is meaningless — the fp32 denominator is float noise, so the ratio
    # explodes and says nothing. The informative denominator is what fp16 costs
    # PyTorch on the same input.
    payload: dict = {"coreml": results, "torch_fp16_baseline": torch_fp16_stats}
    if torch_fp16_stats:
        ratio = results["fp16"]["l1_mean"] / max(torch_fp16_stats["l1_mean"], 1e-12)
        payload["coreml_fp16_over_torch_fp16_l1_ratio"] = ratio
        print(f"\ncoreml fp16 L1 {results['fp16']['l1_mean']:.6f}  /  "
              f"torch fp16 L1 {torch_fp16_stats['l1_mean']:.6f}  =  {ratio:.2f}x")
        print("(issue #2625 reports 0.0394 / 0.0165 = 2.39x)")

    (out / f"{args.model}.drift.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.model}.drift.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
