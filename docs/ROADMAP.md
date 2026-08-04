# Roadmap

Build order and current state. Design rationale is in [`SPEC.md`](SPEC.md);
what we validate and why is in [`TEST_STRATEGY.md`](TEST_STRATEGY.md).

The sequencing principle: **experiments that produce measurements come before
infrastructure that reports them.** Thresholds in `TEST_STRATEGY.md` are derived
from measured data, so the measurements have to exist first — each remaining
⟨measure⟩ marker is a dependency on an unfinished milestone below.

---

## Environment of record

Every threshold is only meaningful relative to this configuration, and none of
it is assumed portable across chips (`TEST_STRATEGY.md` §4.4).

| | |
|---|---|
| Chip | Apple **M4 Pro** |
| macOS | **26.5.2** (build 25F84) |
| Xcode | **26.6** (17F113), macOS 26.5 SDK |
| Swift | **6.3.3**, Swift Testing 1902 |
| Python | dedicated conda env, **3.12** |

### Toolchain notes

Two things bite on a fresh machine, both recorded in
[`FINDINGS.md`](FINDINGS.md) F-001/F-002:

**`xcode-select` may point at CommandLineTools**, in which case `xcodebuild`
fails even with Xcode installed:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
# or, without sudo:
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

**Python 3.14 is ahead of torch/coremltools support.** Use a pinned 3.12 env;
do not install into the system or conda `base` interpreter:

```bash
conda create -n inferenceprobe python=3.12 -y
conda activate inferenceprobe
pip install torch torchvision coremltools numpy pillow
pip freeze > python/requirements.txt   # pinning is mandatory — SPEC §5
```

The environment already exists on the machine of record; normal startup does
not recreate or reinstall it:

```bash
cd /Users/weihantu/InferenceProbe/python
conda activate inferenceprobe
python --version                       # expected: 3.12.13
```

If a new zsh session does not recognize `conda activate`, run
`source /Users/weihantu/miniforge3/etc/profile.d/conda.sh` first. The separate
`inferenceprobe-torch27` environment is a controlled comparison for F-010, not
the normal development environment. Run `conda deactivate` when finished.

---

## Milestones

### M0 — Harness skeleton · ✅ done

`Package.swift`, `HostEnvironment` with chip detection, first Swift Testing
suite green. Establishes that every measurement can be stamped with the machine
that produced it.

### M1 — Export and reference · ✅ done

`python/export_model.py` exports MobileNetV3-Large to Core ML at **fp32 and
fp16**, predicts through each on `.cpuOnly`, and writes a per-artifact manifest
recording resolved versions and host chip. Both precisions matter: coremltools
defaults to fp16, so the fp32 export is what makes the fp16 delta attributable
(SPEC §2.4).

Produced two findings: a native crash in `ct.convert()` (F-005), later traced to
GIL mishandling and worked around, and a first drift measurement (F-006, n=1,
preliminary).

### M2 — The two core experiments · ✅ done

1. **fp16 drift vs. the PyTorch reference.** Measure L1 against the PyTorch
   fp32→fp16 baseline. Reference figures from
   [coremltools #2625](https://github.com/apple/coremltools/issues/2625):
   **0.0394 vs. 0.0165**. That issue was filed on M1 hardware / macOS 26.1, so a
   non-reproduction on M4 Pro is a legitimate result and evidence for the
   hardware-dependence claim in the thread.
2. **Compute-unit differential.** Same model under all four compute units,
   disagreement measured against the `.cpuOnly` anchor.

Experiment 2 does not depend on that issue existing, and is the more valuable of
the two. It is not gated on experiment 1 succeeding.

**Result.** The fp16 direction reproduced, but at **1.51–1.70×** the PyTorch
fp16 baseline across two synthetic input distributions rather than #2625's
2.39×. Input distribution and torch version were eliminated as explanations;
hardware/OS remains the leading explanation but cannot be proven on this
machine (F-006, F-009, F-010). The compute-unit differential ran over 200
inputs: every path had a zero repeat-run noise floor, while CPU-vs-ANE changed
top-1 on 3/200 synthetic inputs (F-008).

### M3 — Localization and variance · ⏳ partial

- ⏳ `MLModelComparator` op localization **swept across `atol`**, to find where
  the answer stabilizes (`TEST_STRATEGY.md` §4.5).
- ⏳ `MLComputePlan` → ANE residency, setting the §5.3 threshold.
- ✅ **Run-to-run variance on a fixed input**, per compute unit: zero maximum
  spread across 20 repeats for all four units (F-008).
- ⏳ `MLModelValidator` NaN/Inf scan, closing the last pending Day-3
  investigation.

### M4 — Swift test suite · next

`ModelRunner`, `TolerancePolicy`, `GoldenStore`. The model-validation targets
from SPEC §3 — `ParityTests`, `ComputeUnitTests`, `MetamorphicTests`,
`RobustnessTests`, `PerformanceTests` — land here. Failure messages must name the
offending layer, not report that outputs differ.

### M5 — CI · not started

GitHub Actions on `macos-15`, PR gate under the 5-minute budget from
`TEST_STRATEGY.md` §6, `.xcresult` posted as a readable PR check. Uses
`xcodebuild`, not `swift test` — only the former emits the result bundle the
reporting tools consume (SPEC §2.1).

---

## Backlog

- **On-device validation over `devicectl`** — coremltools' Remote-Device utility
  extends every parity and perf test to real silicon. The largest single gap,
  given the project's premise; needs an Apple Developer Team ID.
- **Model #2 (DistilBERT), then Whisper-tiny.** A second modality is what proves
  the harness is model-agnostic — the difference between a framework and a
  script. Whisper adds autoregressive decoding, the richest test surface.
- **Test plans + XCUITest**, once an Xcode project exists for the demo app.
- **Performance baselines** committed, activating the regression gate (SPEC §4.8).
- Consider contributing the `atol` sweep upstream to coremltools #2625.
