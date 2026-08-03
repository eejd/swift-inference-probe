#!/usr/bin/env python3
"""Reproduce coremltools issue #2625 — and test whether the number is stable.

Issue #2625 reports MobileNetV3-Large converting to fp16 Core ML with L1 drift
0.0394 against a PyTorch fp32->fp16 baseline of 0.0165 — a ratio of ~2.39x.

F-006 measured 1.78x here, on n=1 with uniform random pixels. Three confounds
separate this machine from the issue: chip (M4 Pro vs M1), torch version
(2.13.0 vs 2.7.0), and *input distribution*. This script eliminates the third,
and in doing so tests something more interesting than the reproduction itself:

    is the reported ratio a property of the model, or of the inputs it was
    measured on?

White noise is off-distribution for a conv net trained on photographs. Natural
images have a 1/f^alpha power spectrum — smooth gradients, correlated
neighbouring pixels, structure at every scale. A network's activations, and
therefore its logit magnitudes, differ substantially between the two. L1 is not
scale-free, so if logit magnitudes move, L1 moves with them.

Both distributions are run so the comparison is visible rather than assumed.

Usage:
    python repro_2625.py                    # n=100 per distribution
    python repro_2625.py --n 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

import coremltools as ct

from export_model import drift  # same statistics as the exporter, by construction

SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
INPUT_NAME = "input"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def white_noise(rng: np.random.Generator) -> np.ndarray:
    """Uniform pixels. What F-006 used. Off-distribution for a photo model."""
    return rng.uniform(0.0, 1.0, (3, 224, 224))


def natural_noise(rng: np.random.Generator, alpha: float = 1.2) -> np.ndarray:
    """Synthetic image with a 1/f^alpha power spectrum.

    Not a photograph, but it carries the spatial statistic that matters here:
    energy concentrated at low frequencies, so neighbouring pixels correlate and
    the image has structure at multiple scales. Colour channels share a common
    luminance component with smaller independent variation, as real images do.
    """
    h, w = 224, 224
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fy ** 2 + fx ** 2)
    freq[0, 0] = 1e-6                      # avoid a divide-by-zero at DC
    amplitude = 1.0 / (freq ** alpha)

    def one_plane() -> np.ndarray:
        phase = rng.uniform(0, 2 * np.pi, (h, w))
        plane = np.real(np.fft.ifft2(amplitude * np.exp(1j * phase)))
        lo, hi = plane.min(), plane.max()
        return (plane - lo) / max(hi - lo, 1e-12)

    luminance = one_plane()
    return np.stack([np.clip(0.85 * luminance + 0.15 * one_plane(), 0, 1)
                     for _ in range(3)])


def to_tensor(pixels: np.ndarray) -> torch.Tensor:
    """[0,1] CHW pixels -> ImageNet-normalised NCHW tensor."""
    t = torch.from_numpy(pixels).float().unsqueeze(0)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def summarise(values: list[float]) -> dict:
    a = np.asarray(values)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models-dir", default="../Fixtures/models")
    ap.add_argument("--out", default="../Fixtures/models/repro_2625.json")
    args = ap.parse_args()

    md = Path(args.models_dir)
    print("loading models ...")
    model = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).eval()
    model_fp16 = torchvision.models.mobilenet_v3_large(
        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT).eval().half()
    coreml = {
        p: ct.models.MLModel(str(md / f"mobilenet_v3_large.{p}.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
        for p in ("fp32", "fp16")
    }

    report: dict = {"n": args.n, "issue_2625": {"coreml_fp16_l1": 0.0394,
                                               "torch_fp16_l1": 0.0165,
                                               "ratio": 2.39}}

    for dist_name, generator in (("white_noise", white_noise),
                                 ("natural_1_over_f", natural_noise)):
        rng = np.random.default_rng(SEED)
        torch_fp16_l1, coreml_fp16_l1, coreml_fp32_l1 = [], [], []
        logit_absmax, coreml_fp16_flips = [], 0

        print(f"\n{dist_name}: {args.n} inputs ...")
        for _ in range(args.n):
            x = to_tensor(generator(rng))
            with torch.no_grad():
                ref = model(x).numpy()
                t16 = model_fp16(x.half()).float().numpy()

            c32 = np.asarray(next(iter(coreml["fp32"].predict(
                {INPUT_NAME: x.numpy()}).values())))
            c16 = np.asarray(next(iter(coreml["fp16"].predict(
                {INPUT_NAME: x.numpy()}).values())))

            torch_fp16_l1.append(drift(ref, t16)["l1_mean"])
            coreml_fp32_l1.append(drift(ref, c32)["l1_mean"])
            d16 = drift(ref, c16)
            coreml_fp16_l1.append(d16["l1_mean"])
            coreml_fp16_flips += int(not d16["top1_agrees"])
            logit_absmax.append(float(np.abs(ref).max()))

        ratio = float(np.mean(coreml_fp16_l1) / np.mean(torch_fp16_l1))
        report[dist_name] = {
            "torch_fp16_l1": summarise(torch_fp16_l1),
            "coreml_fp16_l1": summarise(coreml_fp16_l1),
            "coreml_fp32_l1": summarise(coreml_fp32_l1),
            "ratio_coreml_over_torch": ratio,
            "logit_abs_max": summarise(logit_absmax),
            "coreml_fp16_top1_flips": coreml_fp16_flips,
        }
        print(f"  torch  fp16 L1 {np.mean(torch_fp16_l1):.6f}")
        print(f"  coreml fp16 L1 {np.mean(coreml_fp16_l1):.6f}")
        print(f"  coreml fp32 L1 {np.mean(coreml_fp32_l1):.6f}   (control)")
        print(f"  ratio          {ratio:.2f}x")
        print(f"  logit |max|    {np.mean(logit_absmax):.2f}  "
              f"(scale context: L1 is not scale-free)")
        print(f"  top1 flips     {coreml_fp16_flips}/{args.n}")

    print(f"\n{'distribution':20s} {'ratio':>8s}   vs issue #2625 = 2.39x")
    print("-" * 52)
    for d in ("white_noise", "natural_1_over_f"):
        print(f"{d:20s} {report[d]['ratio_coreml_over_torch']:7.2f}x")

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
