#!/usr/bin/env python3
"""Commit input tensors and their PyTorch fp32 reference outputs.

Every other experiment here regenerates its input from a seed. That works
within one language and breaks across two: F-015 showed Swift and Python agree
to the digit on all-NaN input — the same tensor either way — while their
"seeded random" inputs are entirely different tensors, because `torch.rand` is
a specific generator Swift does not reproduce.

So a cross-language parity test cannot regenerate. It has to read bytes both
sides agree on, which is what this writes.

Format is raw little-endian float32 plus a JSON manifest, not `.npy` as
SPEC 4.6 originally said. `.npy` needs a header parser on the Swift side to buy
nothing; a flat buffer with the shape in the manifest is read with `Data` in
four lines and written with `.tofile()` in one.

Model identity in the manifest is a digest of **weight.bin**, deliberately not
of the `.mlpackage`. F-011 measured the package to be non-reproducible — fresh
UUIDs and an embedded date stamp on every export — so a package digest would
invalidate every golden on every re-export while the numerics were identical.

Usage:
    python generate_goldens.py
    python generate_goldens.py --count 8 --reason "re-blessed after torch bump"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision

import coremltools as ct

from repro_2625 import natural_noise, white_noise, to_tensor

# Repository-relative, not cwd-relative — see export_model.py and F-011.
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "Fixtures" / "models"
INPUTS = ROOT / "Fixtures" / "inputs"
GOLDENS = ROOT / "Fixtures" / "goldens"

SEED = 0
INPUT_SHAPE = [1, 3, 224, 224]
OUTPUT_SHAPE = [1, 1000]


def host_chip() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"


def weight_digest(precision: str) -> str | None:
    """sha256 of the model weights — stable across exports, unlike the package."""
    path = (MODELS / f"mobilenet_v3_large.{precision}.mlpackage"
            / "Data" / "com.apple.CoreML" / "weights" / "weight.bin")
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tensor(path: Path, array: np.ndarray) -> str:
    """Raw little-endian float32. Returns the sha256 of what was written."""
    flat = np.ascontiguousarray(array, dtype="<f4")
    path.write_bytes(flat.tobytes())
    return hashlib.sha256(flat.tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 2 per distribution is what is committed, so the bare command reproduces
    # the tracked state rather than adding to it. Volume lives in the Python
    # experiments, where n=100-200 costs nothing; these fixtures exist for
    # cross-language *exactness* (F-015), which needs precision not quantity.
    ap.add_argument("--count", type=int, default=2,
                    help="inputs per distribution (each is ~590 KB on disk)")
    ap.add_argument("--reason", default="initial baseline",
                    help="why these goldens are being written — recorded in the manifest")
    args = ap.parse_args()

    INPUTS.mkdir(parents=True, exist_ok=True)
    GOLDENS.mkdir(parents=True, exist_ok=True)

    print("loading pytorch reference ...")
    model = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).eval()

    digests = {p: weight_digest(p) for p in ("fp32", "fp16")}
    if not all(digests.values()):
        sys.exit("missing .mlpackage weights — run export_model.py first")

    produced_by = {
        "chip": host_chip(),
        "macos": platform.mac_ver()[0],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
    }

    written = []
    # Two distributions, matching F-009: white noise is off-distribution for a
    # photograph model, and the 1/f set carries the spatial statistic that
    # distinguishes real images. A parity suite that only ever sees one is
    # measuring one operating point.
    for dist_name, generator in (("white_noise", white_noise),
                                 ("natural_1_over_f", natural_noise)):
        rng = np.random.default_rng(SEED)
        for index in range(args.count):
            identifier = f"mobilenetv3_large.{dist_name}_{index:03d}"
            x = to_tensor(generator(rng))

            with torch.no_grad():
                reference = model(x).numpy()

            input_path = INPUTS / f"{identifier}.input.f32.bin"
            golden_path = GOLDENS / f"{identifier}.pytorch_fp32.f32.bin"
            input_sha = write_tensor(input_path, x.numpy())
            golden_sha = write_tensor(golden_path, reference)

            manifest = {
                "id": identifier,
                "distribution": dist_name,
                "producedBy": produced_by,
                "blessedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": args.reason,
                "input": {"file": input_path.name, "shape": INPUT_SHAPE,
                          "dtype": "float32", "sha256": input_sha},
                "reference": {"file": golden_path.name, "shape": OUTPUT_SHAPE,
                              "dtype": "float32", "sha256": golden_sha,
                              "source": "pytorch fp32 (torchvision pretrained)"},
                # F-011: identity is the weights, never the .mlpackage.
                "modelWeightSha256": digests,
                "top1": int(reference.argmax()),
            }
            (GOLDENS / f"{identifier}.json").write_text(
                json.dumps(manifest, indent=2) + "\n")
            written.append((identifier, int(reference.argmax()),
                            float(np.abs(reference).max())))
            print(f"  {identifier}  top1={manifest['top1']:<4d} "
                  f"|max|={np.abs(reference).max():.4f}")

    index = {
        "count": len(written),
        "ids": [name for name, _t, _m in written],
        "format": "raw little-endian float32; shape and sha256 in each manifest",
        "note": ("Regenerating inputs from a seed is not reproducible across "
                 "languages (F-015). These bytes are the contract."),
    }
    (GOLDENS / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {len(written)} goldens to {GOLDENS}")
    print(f"      {len(written)} inputs  to {INPUTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
