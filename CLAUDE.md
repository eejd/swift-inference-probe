# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A test harness validating Core ML inference on Apple silicon for numerical
correctness across compute units, robustness, and performance regression.

**Current state as of 3 Aug 2026:** M0 (Swift harness skeleton), M1 (fp32/fp16
export), and M2 (the two core numerical experiments) are complete. M3 is
partial: the run-to-run noise floor is measured, while op localization, compute
plan residency, and NaN/Inf validation remain. The reusable Swift harness and CI
from M4/M5 are not built yet. `Sources/` therefore contains only
`HostEnvironment`, while `python/` contains the working experiment scripts.

The design lives in `docs/`, and those documents are the spec — read them before
writing code, and update them when a decision changes.

| Document | Role |
|---|---|
| `docs/SPEC.md` | Architecture, component contracts, key decisions with rationale |
| `docs/TEST_STRATEGY.md` | What we validate and why; the tolerance policy; risk tiers |
| `docs/ROADMAP.md` | Build order, environment of record, backlog |
| `docs/FINDINGS.md` | Every discrepancy found, with a judgement on each |
| `docs/TESTING_PRIMER.md` | Why ML inference needs different testing tools; the concepts behind the design |

Anything matching `docs/*.private.md`, plus `apple-ml-test-automation-playbook.md`,
is personal and **gitignored**. Do not commit it, quote it, or copy its contents
into a tracked file, and they must
not be published. `docs/` is authoritative for all engineering decisions.

## Environment gotchas

Both verified on this machine, and both will waste an hour if rediscovered.

**A fresh shell may have `xcode-select` pointed at CommandLineTools, which makes
`xcodebuild` fail** even though Xcode 26.6 is installed. This machine was fixed
on 2 Aug; if the skew returns, either fix it once (`sudo xcode-select -s
/Applications/Xcode.app/Contents/Developer`) or prefix commands with
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, which needs no sudo.
`XCTest.framework` exists only under the Xcode platform directory, never in the
CommandLineTools SDK.

**Default `python3` is Homebrew 3.14.2**, which is ahead of torch/coremltools
support. Use the dedicated conda env; never install ML packages into the default
interpreter or conda `base`.

```bash
conda activate inferenceprobe   # python 3.12
```

Host is **M4 Pro / macOS 26.5.2**. This matters: tolerances are chip-specific by
design (`TEST_STRATEGY.md` §4.4), and the upstream issue this project
investigates was filed on M1 hardware.

## Commands

The package currently contains the `InferenceProbe` library and `HarnessTests`.
The larger validation target names in `docs/SPEC.md` are planned for M4. The
runner invocations below are verified working on this machine (FINDINGS F-003).

Local iteration — fast, no result bundle:

```bash
swift test                                         # whole suite
swift test --filter HostEnvironment                # current suite
```

CI — **use `xcodebuild`, not `swift test`**, because only it emits the
`.xcresult` that `xcresulttool`, `xcresultparser`, and `flake-radar` consume:

```bash
xcodebuild test -scheme InferenceProbe-Package -destination 'platform=macOS' \
  -resultBundlePath result.xcresult \
  -retry-tests-on-failure -test-iterations 3

xcrun xcresulttool get test-results metrics --path result.xcresult   # raw per-iteration data
```

**The scheme is auto-named `<PackageName>-Package`, not `<PackageName>`.** Using
the wrong name fails with a misleading *"workspace does not contain a scheme"*;
`xcodebuild -list` shows the real one. Also: the legacy `xcresulttool get object
--legacy` returns zero performance metrics — use `get test-results metrics`.

`.xctestplan` files need an Xcode project, but the flags above (`-only-testing:`,
`-skip-testing:`, `-retry-tests-on-failure`, `-run-tests-until-failure`,
`-parallel-testing-enabled`) cover what a test plan would do. What is genuinely
lost is *committed, reviewable* configuration — it lives in workflow YAML
instead. Keep Swift Testing tags aligned with `TEST_STRATEGY.md` §6 so migrating
to test plans later is mechanical.

### Starting the Python environment

Run the scripts from `python/`, because their default input and output paths are
relative to that directory:

```bash
cd /Users/weihantu/InferenceProbe/python
conda activate inferenceprobe
python --version
python -c 'import torch, torchvision, coremltools; print(torch.__version__, torchvision.__version__, coremltools.__version__)'
```

Expected primary-environment versions are Python 3.12.13, torch 2.13.0,
torchvision 0.28.0, and coremltools 9.0. If `conda activate` says the shell is
not initialized, load conda for the current terminal and retry:

```bash
source /Users/weihantu/miniforge3/etc/profile.d/conda.sh
conda activate inferenceprobe
```

coremltools will warn that torch 2.13.0 is newer than the version it has tested.
That warning is expected, not an activation failure; F-010 repeats the
measurements under the supported torch 2.7.0 pairing.

`conda init zsh` makes that shell setup persistent for future terminals. Use
`conda deactivate` when finished. The comparison environment from F-010 is
activated with `conda activate inferenceprobe-torch27`; it has torch 2.7.0 and
torchvision 0.22.0 and should not replace the primary environment.

After activation, the existing experiments are:

```bash
python export_model.py
python compute_unit_differential.py
python repro_2625.py
```

Run `pip freeze > requirements.txt` from `python/` after any intentional primary
environment dependency change, because the coremltools debugging utilities are
experimental and version-sensitive.

## Architecture

### The organizing idea: there is no oracle

ML inference has no known-correct expected value to assert against — different
compute paths can produce systematically different floating-point outputs, and
coremltools converts to fp16 by default, so divergence from the PyTorch fp32
reference is *expected behaviour*. On this M4 Pro, each individual compute path
is bit-deterministic across repeated runs (F-008); that does not make outputs
bit-identical *between* CPU, GPU, and ANE. Writing `assertEqual` across those
paths still asserts something false.

Everything in the codebase follows from replacing it with three mechanisms:
**differential testing** (the four compute units must agree, so the runtime is
its own oracle), **golden tests with an explicit tolerance policy** (assert on
drift, not bit-equality), and **metamorphic relations** (properties between
related inputs, needing no ground truth). If a change does not fit one of those
three, it probably belongs somewhere else.

### Invariants worth knowing before editing

- **`.cpuOnly` is the reference anchor.** Compare every other compute unit
  *against it*, never pairwise-all — O(n) instead of O(n²), and failures get a
  single clear referent. It is privileged because it is the simplest, most
  stable dispatch path, not because it is uniquely deterministic; F-008 found
  all four paths deterministic on this machine.
- **`TolerancePolicy.rationale` is a required, non-optional `String`.** A policy
  cannot be constructed without stating why those thresholds. This is
  intentional: it makes the documentation a type-system property rather than a
  review convention. Do not make it optional.
- **`GoldenStore.bless(_:as:reason:)` requires a reason**, recorded with the
  environment that produced it. Blessing is an audited act, not a `--fix` flag.
- **Thresholds are measured, not chosen.** `TEST_STRATEGY.md` §4.3 records the
  measured GPU/ANE differential thresholds and leaves unresolved ones marked
  ⟨measure⟩. Do not substitute a round number, and never widen a threshold to
  make a test green — that is the specific failure this process exists to
  prevent.
- **Every result carries `HostEnvironment`.** A golden blessed on one chip is not
  assumed valid on another.

### Why two test frameworks

Swift Testing is the default for unit tests but covers neither UI automation nor
performance measurement. `XCUITest` and the `XCTMetric` family stay in XCTest,
and both coexist in the same target. Parity, compute-unit, metamorphic, and
robustness tests are Swift Testing; performance and UI tests are XCTest. See
`SPEC.md` §2.2 for the per-target table.

`XCTest`'s `measure()` reports mean and standard deviation, **not percentiles** —
hence the custom p50/p95 harness in `Metrics.swift`. That is a deliberate
addition, not duplication.

### Why there is a Python half

Apple's op-level debugging utilities — `MLModelComparator`, `MLModelValidator`,
`MLModelInspector`, `MLModelBenchmarker` — are Python-only and live under
`coremltools.models.ml_program.experimental`. So Swift asserts, and Python
produces references and localizes failures to a specific op or torch module. A
parity failure should name the offending layer; that name comes from the Python
side.

Note that **op localization is itself tolerance-dependent** — `MLModelComparator`
flags different layers depending on the `atol` passed. Localization results are
meaningless without the `atol` that produced them.

### SwiftPM is the source of truth

`Package.swift` defines the library and all non-app-hosted test targets. The old
zero-target `InferenceProbe.xcodeproj` was removed because it shadowed the
package during `xcodebuild` discovery (F-004), and the path is ignored so it
cannot accidentally return. A real project will be created when the SwiftUI
demo app is added post-deadline.
