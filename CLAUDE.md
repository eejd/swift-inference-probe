# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A test harness validating Core ML inference on Apple silicon for numerical
correctness across compute units, robustness, and performance regression.

**The repository is pre-implementation as of 1 Aug 2026.** `Sources/`, `Tests/`,
and `python/` are empty skeletons; there is no `Package.swift` yet. The design
lives in `docs/`, and those documents are the spec — read them before writing
code, and update them when a decision changes.

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

**`xcode-select` points at CommandLineTools, so `xcodebuild` fails** even though
Xcode 26.6 is installed. Either fix it once (`sudo xcode-select -s
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

No package exists yet, so target names are from `docs/SPEC.md`. The **runner
invocations below are verified working** on this machine (FINDINGS F-003).

Local iteration — fast, no result bundle:

```bash
swift test                                         # whole suite
swift test --filter ComputeUnitTests               # one target
swift test --filter testANEParityWithinTolerance   # one test
```

CI — **use `xcodebuild`, not `swift test`**, because only it emits the
`.xcresult` that `xcresulttool`, `xcresultparser`, and `flake-radar` consume:

```bash
xcodebuild test -scheme InferenceProbe-Package -destination 'platform=macOS' \
  -resultBundlePath result.xcresult \
  -only-testing:ParityTests -retry-tests-on-failure -test-iterations 3

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

Python side runs from `python/` inside the conda env; `pip freeze >
python/requirements.txt` after any dependency change, because the coremltools
debugging utilities are experimental and version-sensitive.

## Architecture

### The organizing idea: there is no oracle

ML inference has no known-correct expected value to assert against — GPU and ANE
execution is not bit-reproducible, and coremltools converts to fp16 by default,
so divergence from the PyTorch fp32 reference is *expected behaviour*. Writing
`assertEqual` here asserts something false.

Everything in the codebase follows from replacing it with three mechanisms:
**differential testing** (the four compute units must agree, so the runtime is
its own oracle), **golden tests with an explicit tolerance policy** (assert on
drift, not bit-equality), and **metamorphic relations** (properties between
related inputs, needing no ground truth). If a change does not fit one of those
three, it probably belongs somewhere else.

### Invariants worth knowing before editing

- **`.cpuOnly` is the deterministic anchor.** Compare every other compute unit
  *against it*, never pairwise-all — O(n) instead of O(n²), and failures get a
  single clear referent.
- **`TolerancePolicy.rationale` is a required, non-optional `String`.** A policy
  cannot be constructed without stating why those thresholds. This is
  intentional: it makes the documentation a type-system property rather than a
  review convention. Do not make it optional.
- **`GoldenStore.bless(_:as:reason:)` requires a reason**, recorded with the
  environment that produced it. Blessing is an audited act, not a `--fix` flag.
- **Thresholds are measured, not chosen.** `TEST_STRATEGY.md` §4.3 marks unset
  ones ⟨measure⟩; they are filled from run-to-run variance. Do not substitute a
  round number, and never widen a threshold to make a test green — that is the
  specific failure this process exists to prevent.
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

`Package.swift` defines the library and all non-app-hosted test targets. The
existing `InferenceProbe.xcodeproj` is an **empty shell with zero targets** that
references only the playbook markdown — it is not load-bearing and will be
regenerated when the SwiftUI demo app is added post-deadline.
