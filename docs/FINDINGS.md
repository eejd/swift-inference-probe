# Findings

Every real discrepancy the harness caught, with the layer, the magnitude, the
compute path, and a judgement on whether it is acceptable.

**As of 1 Aug 2026 there are no inference findings yet** — the harness does not
run. What follows is the environment record, without which no later number means
anything, plus the toolchain findings from the Day-1 gate.

Entry format: what was observed · how it was measured · **judgement**. A finding
without a judgement is an observation, not a finding.

---

## Environment of record

Every measurement in this document is relative to this configuration. Goldens
are stamped with it (SPEC §4.6) and it is *not* assumed portable across chips —
see [`TEST_STRATEGY.md`](TEST_STRATEGY.md) §4.4.

| | |
|---|---|
| Chip | Apple **M4 Pro** |
| macOS | **26.5.2** (build 25F84) |
| Xcode | **26.6** (17F113) |
| macOS SDK | 26.5 |
| Swift | **6.3.3** (swiftlang-6.3.3.1.3), Swift Testing 1902 |
| Python | **3.12.13** (conda env `inferenceprobe`) |
| torch | **2.13.0** |
| torchvision | **0.28.0** |
| coremltools | **9.0** |
| numpy | **2.5.1** |

Full pin in `python/requirements.txt`. Note **coremltools 9.0 warns that torch
2.7.0 is the newest version it has been tested against**, so torch 2.13.0 is an
unvalidated pairing. That is the same coremltools major as issue #2625, but a
different torch — see F-005 and the reproduction caveats in F-006.

**Comparison target.** coremltools [#2625](https://github.com/apple/coremltools/issues/2625)
was filed on **M1 Air + M1 Max / macOS 26.1 / coremltools 9.0 / PyTorch 2.7**.
This machine differs in **both chip generation and OS version**. The issue thread
already claims the divergence is hardware-dependent, so this configuration is a
third data point whichever way the repro goes.

---

## F-001 · `xcode-select` pointed at CommandLineTools, not Xcode

**Observed.** `xcodebuild -version` failed with *"tool 'xcodebuild' requires
Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a
command line tools instance"* — despite Xcode 26.6 being installed at
`/Applications/Xcode.app`.

**Measured.** `xcode-select -p` → `/Library/Developer/CommandLineTools`.
Confirmed the Xcode toolchain is otherwise intact:
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -version`
→ `Xcode 26.6 / Build version 17F113`. `XCTest.framework` is present under the
Xcode platform directory and **absent** from the CommandLineTools SDK.

**Judgement.** Environment, not product. Worth recording for two reasons: it
would have blocked every XCTest and `xcodebuild` invocation later in the week,
and it is precisely the kind of silent environment skew that makes CI results
non-comparable to local ones.

The skew was not only about `xcodebuild` availability. The two toolchains ship
**different compilers** — CommandLineTools had Swift **6.0**, Xcode 26.6 has
**6.3.3** — so before the fix, `swift build` and
`DEVELOPER_DIR=… xcrun swift build` compiled the same source with different
versions. That is exactly the class of difference that makes a local green run
and a red CI run irreconcilable, and it is invisible unless you check.

**Resolution: fixed** 2 Aug via
`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`. Verified:
`xcode-select -p` resolves to Xcode, `xcodebuild -version` succeeds with no
`DEVELOPER_DIR`, `swift --version` reports 6.3.3, and `XCTest.framework` is
reachable in the active developer directory. All measurements in this document
are now attributable to a single toolchain.

---

## F-002 · Default interpreter is ahead of the ML stack

**Observed.** `python3` on `PATH` resolves to Homebrew **3.14.2**. `torch`,
`coremltools`, and `mlx` import-fail in every interpreter on the machine.

**Measured.** Interpreters found: Homebrew 3.14.2 (default), conda base 3.12.7,
python.org framework 3.12.5, system 3.9. No `torch` or `coremltools` in any.

**Judgement.** Would have surfaced as a confusing pip resolution failure mid-Day-1
had the gate not run first. Resolved by pinning a dedicated conda env at 3.12
rather than installing into `base` — see [`ROADMAP.md`](ROADMAP.md) §Environment. This
is also *why* `requirements.txt` is pinned rather than floating: the experimental
coremltools debugging utilities this project depends on are version-sensitive
(SPEC §5). **Resolution: pending.**

---

## F-003 · SwiftPM vs. Xcode project — measured, not assumed

**Question.** `SPEC.md` §2.1 originally specified `swift test` for CI on the
assumption that an `.xcodeproj` bought us little before 7 Aug. Rather than assume,
a throwaway package was built with one Swift Testing target and one XCTest target
using `measure(metrics:)`, and both runners were tried.

**Measured** on Xcode 26.6 (17F113) / macOS 26.5.2 / M4 Pro, Swift 6.3.3,
Swift Testing 1902:

| Question | Result |
|---|---|
| Swift Testing + XCTest in one package under `swift test`? | ✅ both ran; parameterized `@Test(arguments:)` produced 3 cases |
| `measure(metrics:[XCTClockMetric, XCTMemoryMetric, XCTCPUMetric])` under `swift test`? | ✅ ran, emitted 6 metrics |
| Performance **baseline** comparison under `swift test`? | ❌ `baselineName: ""`, `baselineAverage: 0` — gate inert |
| `xcodebuild test` against a **bare `Package.swift`**? | ✅ `** TEST SUCCEEDED **`, 5 tests |
| `.xcresult` bundle produced? | ✅ 256 KB |
| Raw per-iteration perf data recoverable? | ✅ `xcresulttool get test-results metrics` returns `measurements` arrays |
| Retry / selection / parallel flags without a test plan? | ✅ `-retry-tests-on-failure`, `-test-iterations`, `-run-tests-until-failure`, `-only-testing:`, `-skip-testing:`, `-parallel-testing-enabled` |

**Two gotchas.** The implicit scheme is named **`<PackageName>-Package`**, not
`<PackageName>` — `xcodebuild -list` to discover it; using the wrong name fails
with a misleading *"workspace does not contain a scheme"*. And the **legacy**
`xcresulttool get object --legacy --format json` returns **zero** performance
metrics; the current `get test-results metrics` subcommand is required.

**Judgement.** The original decision was **wrong in one respect and is
corrected**: CI now uses `xcodebuild`, because `swift test` yields no `.xcresult`
and every reporting tool downstream consumes one. The rest of the SwiftPM
decision stands — an `.xcodeproj` is not needed before 7 Aug.

Usefully, the `.xcresult` also auto-stamps device identity (architecture, model,
OS build), which corroborates the `HostEnvironment` design in SPEC §4.1, and the
raw `measurements` arrays mean percentiles are computed in post-processing rather
than hand-rolled. **Resolution: SPEC §2.1, §4.8 and TEST_STRATEGY §7, §8 updated.**

---

## F-004 · An empty `.xcodeproj` silently shadows `Package.swift`

**Observed.** With `xcode-select` fixed (F-001), `xcodebuild test -scheme
InferenceProbe-Package` failed:

> `xcodebuild: error: The project named "InferenceProbe" does not contain a
> scheme named "InferenceProbe-Package".`

The message points at the scheme name. The scheme name was not the problem.

**Measured.** `xcodebuild -list` in the package root reported *"There are no
targets in project «InferenceProbe». This project contains no schemes"* — so
`xcodebuild` had resolved the leftover **zero-target `InferenceProbe.xcodeproj`**
and never looked at `Package.swift` at all. Moving the project bundle out of the
directory and re-running:

```
Information about workspace "InferenceProbe":
    Schemes:
        InferenceProbe-Package
** TEST SUCCEEDED **     # 8 test cases, .xcresult readable
```

**Judgement.** **When a `.xcodeproj` is present in the package root,
`xcodebuild` uses it and ignores `Package.swift`.** Not a warning — a hard
failure whose error message misdirects toward the scheme argument. This directly
threatens the CI design in SPEC §2.1, which depends on `xcodebuild` driving the
package, and it would have failed on Wednesday with a message that sends you
hunting the wrong thing.

The methodological point is the more interesting one: **F-003 validated this
exact mechanism and missed this, because the spike ran in a clean scratchpad
directory that had no `.xcodeproj`.** The spike environment differed from the
real environment in the one way that mattered, so a passing experiment certified
a workflow that then broke in place. That is a test-validity failure of the same
family the harness exists to catch — a green result answering a slightly
different question than the one asked. Worth remembering when the compute-unit
tests start passing on fixtures that are cleaner than real inputs.

**Resolution: fixed** 2 Aug. The empty project bundle is out of the package root
and untracked; SPEC §2.1 updated to state the constraint rather than call the
project merely "not load-bearing". When the demo app is added post-deadline it
needs a real project with targets, and the `xcodebuild` invocation must then name
the scheme explicitly.

Minor, same session: `-resultBundlePath` **refuses to overwrite** an existing
bundle. Immaterial on a fresh CI runner, but local re-runs need the path cleared
first.

---

## F-005 · `ct.convert()` segfaults if a `predict()` has run in the process

**Observed.** Exporting MobileNetV3-Large at fp32 then fp16 in one process died
with **exit code 139 (SIGSEGV)** during the fp16 conversion. No Python
traceback — a native crash. The fp32 artifact was written; the fp16 artifact
never was.

**Bisected.** The crash is not about fp16, and not about converting twice:

| Sequence in one process | Result |
|---|---|
| `convert(fp16)` alone | ✅ exit 0 |
| `convert(fp32)` → `convert(fp16)` | ✅ exit 0 |
| `convert(fp16)` → `convert(fp32)` | ✅ exit 0 |
| `convert` → `save` → `convert` → `save` | ✅ exit 0 |
| `convert` → `save` → `load` → **`predict`** → `convert` | ❌ **exit 139** |
| `convert(…, skip_model_load=True)` → `save` → `load` → `predict` | ✅ exit 0 |

**Root cause.** `ct.convert()` implicitly loads and compiles the converted model
before returning. On coremltools 9.0 that implicit load **segfaults if any
`predict()` has already run in the same process.** Passing
`skip_model_load=True` suppresses the implicit load and the crash disappears
entirely.

**Judgement.** A real defect, and an unusually nasty one: it is a hard native
crash with no traceback, it depends on an interaction between two calls rather
than on either alone, and the obvious first hypothesis — "fp16 conversion is
broken" — is wrong. A test suite that exports and then predicts in one process
would hit this constantly and blame the wrong component.

Attribution is incomplete: coremltools 9.0 declares torch 2.7.0 as its newest
tested version, and this ran on 2.13.0, so the unvalidated pairing cannot be
excluded as a contributor. Confirming on torch 2.7.0 is the obvious next step
before reporting upstream.

**Resolution: worked around** in `python/export_model.py` via
`skip_model_load=True`, plus an explicit
`ct.models.MLModel(path, compute_units=CPU_ONLY)` load. The workaround is
*better than the original code independently of the bug*: the implicit load uses
`ComputeUnit.ALL`, which is not the deterministic reference path SPEC §2.3
requires, so the explicit load is what should have been there anyway.

---

## F-006 · fp16 drift reproduces in direction, at ~74% of the reported magnitude

**Preliminary — n = 1.** One synthetic input on one machine. Reported here
because it is the first measurement, not because it settles anything.

**Measured.** MobileNetV3-Large, ImageNet-normalised random input, seed 0,
`.cpuOnly`, L1 mean over the 1000 logits:

| Comparison | L1 mean |
|---|---|
| Core ML **fp32** vs PyTorch fp32 | **0.000002** (cos 1.000000, top-1 agrees) |
| PyTorch **fp16** vs PyTorch fp32 — the baseline | **0.006337** |
| Core ML **fp16** vs PyTorch fp32 | **0.011258** (cos 0.999908, top-1 agrees) |
| **Core ML fp16 ÷ PyTorch fp16** | **1.78×** |

Issue #2625 reports **0.0394 / 0.0165 = 2.39×**.

**Judgement.** The *direction* reproduces: Core ML's fp16 conversion drifts
meaningfully more than PyTorch's own fp32→fp16 cast on the same model, which is
the substance of the report. The *magnitude* does not — 1.78× here against 2.39×
there, and both absolute figures are roughly 3× smaller than the issue's.

Three differences could each account for that, and this measurement separates
none of them: **chip** (M4 Pro vs. the M1 Air / M1 Max in the thread, where the
reporter already claims hardware dependence), **torch** (2.13.0 vs 2.7.0), and
**input distribution** (normalised random noise vs. presumably real ImageNet
images — logit magnitudes differ off-distribution, and L1 is not scale-free).

The Core ML **fp32** path is essentially exact (L1 2e-6), which is the useful
control: it says the conversion itself is faithful and the drift is specifically
a precision effect, not a broken graph.

**Do not quote the 1.78× as a reproduction.** With n = 1 on synthetic input it
is a direction, not a magnitude. Making it a real result needs many inputs drawn
from the actual data distribution, and ideally a torch 2.7.0 arm to eliminate one
of the three confounds.

---

## F-007 · Compute-unit choice moves the output more than fp16 conversion does

**Preliminary — n = 1**, same synthetic input and caveats as F-006.

**Measured.** MobileNetV3-Large fp16 `.mlpackage`, one input, each compute unit
compared against the `.cpuOnly` anchor (SPEC §2.3):

| Compute unit | L1 vs `.cpuOnly` | max abs | cosine | top-1 |
|---|---|---|---|---|
| `.cpuAndGPU` | 0.00990185 | 0.09765625 | 0.99993448 | agrees |
| `.cpuAndNE` | 0.01818193 | 0.14453125 | 0.99978100 | agrees |
| `.all` | 0.01818193 | 0.14453125 | 0.99978100 | agrees |

All four units load and predict successfully. Top-1 is stable across all of them.

**Two things worth stating.**

**1. The ANE-vs-CPU disagreement (0.0182) is larger than the entire fp16
conversion drift (0.0113 in F-006).** Which compute unit you run on moved this
model's output *more* than converting it from PyTorch fp32 to Core ML fp16 did.
That reorders the intuition the project started with: fp16 conversion is the
famous problem, but on this model the compute path is the bigger term. It is
also why differential testing across compute units is the higher-value suite —
it targets the larger effect and needs no external reference.

**2. `.all` is bit-identical to `.cpuAndNE`.** Core ML is dispatching this model
to the Neural Engine when given a free choice, so `.all` and `.cpuAndNE` are not
independent test paths *for this model*. That does not make testing both
redundant — the equality is itself the assertion. If `.all` ever diverges from
`.cpuAndNE`, dispatch has changed, and that is exactly the compute-plan
regression TEST_STRATEGY §5.3 is designed to catch. Confirming it against
`MLComputePlan` op assignments is the M3 task.

**Judgement.** Nothing is broken — this is a characterisation, not a defect. Its
value is that it sets up the real experiment: the thresholds in TEST_STRATEGY
§4.3 must sit above run-to-run variance, and this says the ANE gate needs to
tolerate roughly 2× what the GPU gate does. Both numbers are n=1 and cannot be
turned into thresholds until the M3 variance study runs.

---

## Pending investigations

Each resolves to a finding entry, including the negative results — *"could not
reproduce on this configuration"* is a finding, and given the hardware-dependence
claim in #2625, an informative one.

| # | Question | Day |
|---|---|---|
| 1 | Does #2625's fp16 L1 drift (0.0394 vs. 0.0165) reproduce on M4 Pro / macOS 26.5.2? | 2 |
| 2 | What fraction of inputs disagree between `.cpuOnly` and `.cpuAndNeuralEngine`, and by how much? | 2 |
| 3 | Where does `MLModelComparator` op localization stabilize under an `atol` sweep? | 3 |
| 4 | Run-to-run variance on a fixed input, per compute unit — the basis for every threshold | 3 |
| 5 | What ANE residency does MobileNetV3-Large actually achieve? | 3 |
| 6 | Any NaN/Inf-producing ops via `MLModelValidator`? | 3 |
