# Findings

Every real discrepancy the harness caught, with the layer, the magnitude, the
compute path, and a judgement on whether it is acceptable.

**As of 4 Aug 2026 the repository has thirteen findings.** F-001–F-004 establish
the toolchain and runner, F-005 isolates a coremltools native crash, F-006–F-010
record the fp16 and compute-unit experiments, F-011 establishes that those
experiments reproduce byte-for-byte from a fresh export, F-012 localizes both
the precision and compute-unit divergence to a specific op, and F-013 shows the
NaN/Inf gate is blind to the failure mode that actually occurs. The reusable
Swift validation harness is still largely unimplemented, but the Python
experiments run and their result JSON is committed.

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
(SPEC §5). **Resolution: fixed.** The primary `inferenceprobe` environment is
Python 3.12.13 with the exact versions in `python/requirements.txt`; the isolated
`inferenceprobe-torch27` environment exists only for F-010's version-comparison
arm.

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

**Attribution resolved 3 Aug — this is a genuine coremltools defect.** The
original write-up could not exclude the untested torch 2.13.0 pairing. Re-running
the identical sequence in a clean env on **torch 2.7.0** — the version
coremltools 9.0 declares as its newest tested, and the version #2625 was filed
against — **still crashes**.

The signal differs, and the torch 2.7.0 run is the more informative one:

| Environment | Exit | Signal |
|---|---|---|
| torch 2.13.0 | 139 | SIGSEGV — raw memory fault, no message |
| torch 2.7.0 | 134 | SIGABRT, with a diagnostic |

```
Fatal Python error: PyEval_SaveThread: the function must be called with the GIL
held, after Python initialization and before Python finalization, but the GIL is
released (the current Python thread state is NULL)
```

**Root cause: GIL mishandling in coremltools' native extension**, not memory
corruption. Something releases the Global Interpreter Lock and then re-enters a
path that assumes it is still held. That explains why the bug is an *interaction*
rather than a property of either call: `predict()` leaves the interpreter's
thread state in a condition that `convert()`'s implicit model-load path then
mishandles.

This is now worth reporting upstream, and the report writes itself: exact
sequence, both signals, the GIL diagnostic, and a five-row bisection table
showing what does *not* trigger it.

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

## F-008 · 1.5% of inputs change top-1 class between CPU and Neural Engine

**n = 200 inputs, 20 repeats for the noise floor.** `python/compute_unit_differential.py`.

### The noise floor is exactly zero

Same input, same compute unit, 20 consecutive predictions:

| Compute unit | Max spread across 20 runs |
|---|---|
| `.cpuOnly` | **0** — bit-identical |
| `.cpuAndGPU` | **0** — bit-identical |
| `.cpuAndNE` | **0** — bit-identical |
| `.all` | **0** — bit-identical |

**This contradicts the assumption SPEC §2.3 was built on** — that GPU and ANE
execution is not bit-reproducible. On this configuration every compute unit is
perfectly deterministic. SPEC §2.3 has been corrected.

Two consequences, both load-bearing:

1. **All cross-unit disagreement below is systematic, not jitter.** There is no
   noise term to subtract. The measurement is the effect.
2. **A differential test cannot flake for numerical reasons.** The flake budget
   in TEST_STRATEGY §7 has to worry about timeouts and resource contention, but
   not about float non-determinism. Thresholds can be set tight.

### The differential

200 distinct ImageNet-scaled inputs, each unit against the `.cpuOnly` anchor:

| Unit | L1 mean | L1 p95 | L1 max | cos min | top-1 flips |
|---|---|---|---|---|---|
| `.cpuAndGPU` | 0.010203 | 0.012361 | 0.014912 | 0.999826 | **2 / 200 (1.0%)** |
| `.cpuAndNE` | 0.018858 | 0.021773 | 0.024170 | 0.999574 | **3 / 200 (1.5%)** |
| `.all` | 0.018858 | 0.021773 | 0.024170 | 0.999574 | 3 / 200 (1.5%) |

**Judgement.** **1.5% of inputs produce a different top-1 class depending on
whether the model runs on the CPU or the Neural Engine.** Same model file, same
weights, same input — different answer. On a classifier shipped to a billion
devices, that is a real behavioural difference, not a rounding detail, and it is
invisible to any test that only checks "does inference succeed."

The ordering is coherent and expected: GPU diverges from CPU less than ANE does,
consistent with the Neural Engine using the most aggressive numerics of the
three. Cosine similarity stays above 0.9995 throughout, so the logit vectors are
directionally near-identical — the flips are near-ties reordering, not the model
computing something unrelated. That is the *benign* reading, and it is why top-1
flip rate and cosine belong in the report together: cosine alone would have said
"everything is fine."

`.all` remains bit-identical to `.cpuAndNE` across all 200 inputs, confirming
F-007's inference that Core ML dispatches this model to the Neural Engine given
a free choice.

Incidental: `.cpuAndNE` ran 200 predictions in 0.06s against `.cpuOnly`'s 0.37s,
roughly **6× faster**. The accuracy-vs-latency tradeoff is real and measured, and
it is the actual engineering decision a team would be making here.

**Still to resolve.** These are synthetic inputs. The 1.5% flip rate is a
property of this input distribution, and near-ties are likely more common on
random noise than on real photographs. Re-running on real ImageNet samples is
required before quoting the rate as a general figure.

---

## F-009 · Input distribution moves the #2625 ratio, but does not explain the gap

**n = 100 per distribution.** `python/repro_2625.py`.

F-006 measured 1.78× on n=1 with uniform random pixels and flagged input
distribution as a likely confound: white noise is off-distribution for a network
trained on photographs, and **L1 is not scale-free**, so if logit magnitudes move
the L1 moves with them.

This tests that directly by generating a second input set with a **1/f^1.2 power
spectrum** — the spatial statistic that distinguishes photographs from noise
(energy concentrated at low frequencies, correlated neighbouring pixels,
structure at multiple scales), with colour channels sharing a luminance
component.

| | white noise | natural 1/f | issue #2625 |
|---|---|---|---|
| PyTorch fp16 L1 | 0.006923 | 0.008707 | 0.0165 |
| Core ML fp16 L1 | 0.011749 | 0.013184 | 0.0394 |
| **Ratio** | **1.70×** | **1.51×** | **2.39×** |
| Core ML fp32 L1 (control) | 0.000002 | 0.000002 | — |
| mean logit \|max\| | 5.38 | 6.52 | — |
| top-1 flips | 4/100 | 2/100 | — |

**The manipulation worked.** Mean logit magnitude moved from 5.38 to 6.52, so the
two input sets genuinely put the network at different operating points — this was
a real test, not a null manipulation. Absolute L1 rose for both PyTorch and
Core ML, consistent with L1 tracking output scale.

**But it does not explain the gap, and it argues against the hypothesis.** The
ratio moved from 1.70× to 1.51× — *away* from the reported 2.39×, not toward it.
If more photograph-like inputs lower the ratio, real photographs would be
expected to lower it further. Input distribution is therefore eliminated as the
explanation for the discrepancy with #2625, and eliminated in the stronger
direction: it makes the disagreement worse.

Both absolute figures also remain roughly 2–3× below the issue's (0.0132 vs
0.0394; 0.0087 vs 0.0165), so the difference is not a rescaling of one number —
the whole measurement sits lower here.

**The fp32 control held at 0.000002 under both distributions**, which is what
makes the above interpretable: the conversion is faithful regardless of input, so
everything observed is a precision effect rather than a graph defect.

**Judgement.** A useful negative result. Of the three confounds F-006 named —
chip, torch version, input distribution — the third is now ruled out. **Chip and
torch version remain**, and only one of those is testable on this machine. A
torch 2.7.0 environment (matching both #2625 and coremltools 9.0's declared
tested version) is the next elimination; if the ratio still sits near 1.5×, the
remaining explanation is hardware, which is precisely the claim the issue thread
already makes.

Incidental, consistent with F-008: top-1 flips fell from 4/100 to 2/100 on
structured inputs, supporting the reading that flips are near-ties reordering and
that near-ties are more frequent on off-distribution noise.

---

## F-010 · torch version eliminated; hardware is the only remaining explanation

**A second environment**, `inferenceprobe-torch27`, pins **torch 2.7.0 /
torchvision 0.22.0 / coremltools 9.0** — matching both #2625's configuration and
coremltools 9.0's declared tested version. The original env is untouched, so the
two are directly comparable.

**Every number is bit-identical to torch 2.13.0.** Not close — identical to the
last reported digit, across both input distributions at n = 100 and on the
single-input export:

| | torch 2.13.0 | torch 2.7.0 |
|---|---|---|
| PyTorch fp16 baseline L1 | 0.006337 | 0.006337 |
| Core ML fp32 L1 (control) | 0.000002 | 0.000002 |
| Core ML fp16 L1 | 0.011258 | 0.011258 |
| Core ML fp16 max abs | 0.108827 | 0.108827 |
| white-noise ratio, n=100 | 1.70× | 1.70× |
| natural-1/f ratio, n=100 | 1.51× | 1.51× |

**Judgement.** The torch version has **no effect whatsoever** on the exported
model's numerics here. In hindsight this is the expected result — torch's role
ends at tracing the graph, and the arithmetic is Core ML's — but "expected" is
not "verified," and coremltools' own warning made it a live hypothesis worth
spending twenty minutes on.

### Where the #2625 investigation now stands

F-006 named three candidate explanations for measuring 1.5–1.8× against the
issue's 2.39×. Two are now closed:

| Confound | Status |
|---|---|
| Input distribution | **Eliminated** (F-009) — and it moves the ratio *away* from 2.39× |
| torch version | **Eliminated** (this finding) — bit-identical results |
| **Chip / macOS version** | **Remaining. Not testable on this machine.** |

By elimination, the discrepancy is attributable to **hardware or OS**: M4 Pro /
macOS 26.5.2 here against the M1 Air and M1 Max / macOS 26.1 in the thread.

That is worth stating carefully, because it is the conclusion the issue thread
already argues for from a different direction: a commenter there reports an M1
Air showing high numerical error on ops where an M1 Max is fine. This
investigation reaches the same place independently, by eliminating everything
else, on a third chip generation.

**This is a stronger result than a clean reproduction would have been.** A
reproduction would have confirmed a number someone else already published.
Eliminating two confounds and localising the remaining variance to hardware
adds information to the thread — and it directly supports the design decision in
TEST_STRATEGY §4.4 that tolerance must be a function of the host chip, which was
an inherited assumption until now and is now evidence-backed.

**Caveat.** Elimination is not proof. A fourth unconsidered variable — the
`.mlpackage` deployment target, the Core ML runtime shipped with macOS 26.5.2
versus 26.1, or the specific fixture the issue reporter used — could account for
it. Confirming would need the same script run on an M1, which is the one thing
this machine cannot provide.

---

## F-011 · Export is numerically reproducible; the package around it is not

**Observed.** `export_model.py` was re-run accidentally on 4 Aug, two days after
the export of record and from a different working directory. Comparing the two
exports splits cleanly into two halves: everything numerical is identical, and
everything about the container is not.

**Measured.** Fresh export (4 Aug 04:28 UTC) against the committed export
(2 Aug 08:42 UTC) — same machine, same seed, same `inferenceprobe` environment,
unchanged code:

| Artifact | Result |
|---|---|
| `weight.bin`, fp16 (10,951,440 B) | **byte-identical** |
| `weight.bin`, fp32 (21,892,672 B) | **byte-identical** |
| `mobilenet_v3_large.drift.json` | **byte-identical** |
| `compute_unit_differential.json` | **byte-identical** |
| `repro_2625.json` | **byte-identical** |
| `model.mlmodel` | differs from byte 119; same 116,395 B |
| `Manifest.json` | differs — UUIDs regenerated per export |
| `*.manifest.json` | differs in `exported_at` only, by construction |

The `model.mlmodel` difference is an embedded **local-date stamp**
(`2026-08-02` vs `2026-08-03`) plus metadata key reordering. No weight or op
payload changed.

**Judgement.** Two results, and the second is the actionable one.

**1. Determinism holds far wider than F-008 established.** F-008 measured zero
spread across 20 consecutive predictions *inside one process*. This extends the
claim across separate processes, separate days, and a full re-conversion:
`torch.jit.trace` → `ct.convert` → fp16 quantization is byte-reproducible on a
fixed seed and environment. Every figure in F-006 through F-009 is now
independently reproduced from source rather than re-read from a committed file.

**2. Golden identity must not be a digest of the `.mlpackage`.** The golden
manifest in SPEC §4.6 currently records `producedBy` but no model identity. One
is needed — a golden has to be invalidated when the model genuinely changes —
and the obvious implementation, hashing the `.mlpackage`, is **wrong**: fresh
UUIDs and a date stamp change that hash on every export while the numerics are
identical. It would invalidate every golden on every re-export, and the failure
would look exactly like a real model change. The stable identity is a digest
over `weight.bin` plus a metadata-stripped spec, or over reference outputs
directly. Cheap to know now; a confusing afternoon during M4 otherwise.

**Scope.** Same machine, same seed, same environment, same code. This is
*reproducibility*, not *portability* — it says nothing about whether another
host produces the same weights. That is the F-010 question and it remains open.

**Resolution: cause fixed** 4 Aug. The scripts resolved their default paths
against the current directory, so invoking them from the repository root wrote a
complete 32 MB parallel tree into the repository's *parent*. Defaults now
resolve against `__file__`, so they land in the repository regardless of where
they are invoked from. The stray tree was verified redundant and removed.

---

## F-012 · Op localization is a function of `atol`, and the mechanism is search depth

**n = 1 input, 7 tolerances, 2 comparison arms.** `python/localize.py`, output in
`Fixtures/models/op_localization.json`.

TEST_STRATEGY §4.5 predicted the shape — everything flagged at very low `atol`,
nothing at very high, a plateau between — on the strength of the #2625
reporter's observation that `MLModelComparator` blames ordinary `linear`/`conv`
layers depending on the tolerance passed. Both the shape and the reporter's
observation reproduce, and the sweep also explains *why*.

**Measured.** MobileNetV3-Large, 718 ops (182 non-const), one seeded
ImageNet-scaled input, `rtol` pinned to 0 so only `atol` varies:

| `atol` | precision arm (fp32 vs fp16) | compute arm (`.cpuOnly` vs `.cpuAndNE`) |
|---|---|---|
| 1e-6 | `conv:input_3`, `conv:input_67` | `conv:input_3`, `conv:input_67` |
| 1e-5 | `conv:input_3`, `conv:input_67` | `conv:input_3`, `sigmoid_hard:scale_1` |
| 1e-4 | `conv:input_3`, `conv:input_67` | `conv:input_3`, `sigmoid_hard:scale_1` |
| 1e-3 | `conv:input_3` | `conv:input_3` |
| 1e-2 | `conv:input_3` | `conv:input_3` |
| 1e-1 | **`linear:var_930`** | **`linear:var_930`** |
| 1e0 | *(none)* | *(none)* |

**The reporter's observation, reproduced exactly.** Same model, same input, one
parameter changed: at `atol` 0.1 the tool blames a `linear`; at `atol` ≤ 0.01 it
blames a `conv`. Neither answer is wrong.

**The mechanism.** `find_failing_ops` walks backward from the model outputs and
reports an op as a failure *source* when it fails comparison while its own
inputs pass. So the tolerance decides how far back the walk gets. Op positions
confirm it: `conv:input_3` is non-const op **1 of 182** — the stem convolution —
and `linear:var_930` is op **182 of 182**, the classifier head. At a loose
tolerance only the final accumulation of error is large enough to fail, the
search stops immediately, and the last layer takes the blame. At a strict
tolerance the failure reaches all the way back, and the search descends to the
origin.

Wall-clock is the decisive corroboration, since search depth is exactly what
costs time:

| `atol` | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1e0 |
|---|---|---|---|---|---|---|---|
| precision arm | 63.5s | 62.9s | 62.5s | 62.4s | 52.8s | **0.7s** | 0.3s |
| compute arm | 104.4s | 103.8s | 103.1s | 100.2s | 86.7s | **1.2s** | 0.6s |

The two-orders-of-magnitude collapse at `atol` 0.1 is the search terminating at
the output layer instead of traversing the graph. **`MLModelComparator` is not
misbehaving; it is answering a different question at each tolerance.**

**Where it stabilizes.** Two plateaus per arm, not one:

| Arm | strict plateau | loose plateau | most persistent op |
|---|---|---|---|
| precision | 1e-6…1e-4 → `input_3`, `input_67` | 1e-3…1e-2 → `input_3` | `conv:input_3`, **5/7** points |
| compute | 1e-5…1e-4 → `input_3`, `sigmoid_hard` | 1e-3…1e-2 → `input_3` | `conv:input_3`, **5/7** points |

**Judgement.** Three things, in ascending order of usefulness.

**1. The operating point is `atol` 1e-3 to 1e-2.** Both arms plateau there, both
plateaus name the same single op, and it is the only region where the two
independent comparisons agree exactly. Below it the answer picks up a second op
that differs by arm; above it the answer degenerates to "the last layer".

**2. Both effects originate at the same place.** fp16 conversion error and
CPU-vs-ANE disagreement both localize to `conv:input_3`, the stem convolution,
at 5 of 7 tolerances in each arm. These are different physical causes — reduced
mantissa versus a different execution engine — and they surface at the same op,
which is what a first layer operating on raw normalised input would be expected
to do. The one asymmetry is real and specific: `sigmoid_hard` appears only in
the compute arm, so the Neural Engine's divergence involves an activation that
fp16 conversion alone does not stress.

**3. CLAUDE.md's warning is now demonstrated rather than asserted.** "Localization
results are meaningless without the `atol` that produced them" was inherited
from the issue thread. It is now this repository's own measurement, with the
mechanism identified, and it generalises past this tool: any localizer that
propagates a pass/fail predicate through a graph has its answer set by the
predicate's threshold.

**Caveats, and the second is the one that matters.** `rtol` is pinned to 0, so
the sweep says nothing about relative-tolerance behaviour, where the threshold
would scale with activation magnitude. More importantly this is **n = 1** — one
input, one model. Every figure above could be a property of this input rather
than of the network, and F-008 already showed that per-input behaviour varies
(3/200 inputs flipped top-1). Repeating the sweep across a set of inputs and
reporting how often `conv:input_3` survives is the next step before any of this
is quoted as a general result.

---

## F-013 · Core ML silently sanitizes NaN/Inf input where PyTorch propagates it

**14 inputs × 2 precisions × 2 compute units, plus a PyTorch reference on the
same inputs.** `python/validate.py`, output in
`Fixtures/models/nan_inf_validation.json`.

The task was the Tier 1 detector from TEST_STRATEGY §5.1 — "NaN / Inf in output.
Unconditional failure, no tolerance debate." Running
`find_failing_ops_with_nan_output()` on a nominal input reports zero failing ops
in 0.4 s, which looks like a pass. It is not a pass until the detector is shown
capable of firing, so the battery includes deliberately malformed inputs and a
PyTorch arm.

### The runtimes disagree about what malformed input means

Same tensors, same weights:

| Input | PyTorch fp32 | Core ML fp32 `.cpuOnly` |
|---|---|---|
| all NaN | **1000 NaN**, no top-1 | finite, **top-1 = 904**, \|max\| 6.59 |
| all +Inf | **1000 NaN**, no top-1 | finite, **top-1 = 904**, \|max\| 6.59 |
| 100 NaNs among 150k values | **1000 NaN**, no top-1 | finite, **top-1 = 111**, \|max\| 4.93 |
| nominal | finite, top-1 = 111 | finite, top-1 = 111, \|max\| 5.00 |

PyTorch does the IEEE-correct thing and poisons the output. Core ML returns a
confident, plausible, finite answer. The third row is the sharpest: **injecting
100 NaNs leaves Core ML's prediction unchanged from nominal** — same top-1, a
2% shift in magnitude — while PyTorch declares the result undefined.

Core ML also produces *identical* output for all-NaN and all-Inf input, which is
what sanitization to a common value would look like.

### The detector's coverage is real but narrow

Across all 56 rows the **NaN scan never fired once**. The Inf scan fired
**twice**, both on `fp16` + `.cpuAndNE`, at input scales 1e4 and 1e6 — the only
two configurations in the battery where Core ML emitted a non-finite output at
all.

So the detector is not inert, but it covers a sliver of the failure space. Every
other broken result in this battery is finite, and therefore invisible to it.

### Silent saturation, at a precision-dependent threshold

Scaling the nominal input and comparing top-1 against the fp32 reference:

| input scale | 1e2 | 1e3 | 1e4 | 1e6 | 1e15 | 1e30 | 1e37 |
|---|---|---|---|---|---|---|---|
| fp32 `.cpuOnly` \|max\| | 67.6 | 1217 | 1.23e4 | 1.23e6 | 1.23e15 | 5.99e29 | **3.60** |
| fp16 `.cpuOnly` \|max\| | 68.2 | 632 | **4.56** | 6.52 | 6.52 | 6.52 | 6.52 |
| fp16 top-1 matches fp32 | yes | **no** | no | no | no | no | no |

fp32 tracks the input linearly to about 1e30 and collapses by 1e37. fp16
collapses three decades above nominal, at roughly 1e3, and then **freezes** —
\|max\| pinned at 6.52 and top-1 at 904 for every larger input, the same values
it returns for all-NaN and all-Inf. Neither collapse produces a NaN or an Inf.

### Compute units diverge catastrophically on malformed input

F-008 measured 1.5% top-1 flips between `.cpuOnly` and `.cpuAndNE` on
well-formed input. On malformed input the disagreement is not a flip rate:

| Input | `fp16.cpuOnly` | `fp16.cpuAndNE` |
|---|---|---|
| scale 1e3 | top-1 260, \|max\| 632 | top-1 851, \|max\| 2710 |
| scale 1e4 | top-1 533, \|max\| 4.56, finite | top-1 588, \|max\| 3.18e4, **3 Infs** |
| all +Inf | top-1 904, \|max\| 6.52 | top-1 851, \|max\| 2.99e4 |

Different answers, magnitudes apart by four orders, and one path emits Inf while
the other does not. Incidentally the `fp32` rows are **bit-identical between
`.cpuOnly` and `.cpuAndNE` across all 14 inputs**, which is consistent with the
Neural Engine being fp16-only and the fp32 model never reaching it.

**Judgement.** Three consequences, in ascending order of cost to get wrong.

**1. TEST_STRATEGY §5.1's Tier 1 rule needs qualifying.** "NaN/Inf in output" is
a sound *sufficient* condition for failure and a nearly useless *necessary* one.
On this configuration the dominant failure mode — saturation to a confident
wrong answer — never trips it. The Tier 1 robustness gate needs a finite-but-
implausible check next to it: output magnitude against the reference, or
agreement with the fp32 path, or top-1 stability. A green NaN/Inf scan is not
evidence of a healthy model, and the suite should not present it as one.

**2. A malformed input does not announce itself in production.** A pipeline bug
that feeds NaNs to this model gets a normal-looking class prediction back, not a
crash and not a NaN. PyTorch would have made the same bug obvious in
development. That gap between the training-time and deployment-time runtime is
exactly the class of defect this project exists to find, and it is invisible to
any test that only checks "does inference succeed."

**3. Robustness tests must be differential too.** The single most reliable
signal in this battery was not the validator; it was fp32-vs-fp16 and
CPU-vs-ANE disagreement, which flagged every broken row including the finite
ones. That is the same conclusion F-007 reached from the opposite direction.

**Caveats.** Two, and the first is a genuine open question.

**Where the sanitization happens is not determined by these measurements.** It
could be the coremltools Python wrapper conditioning the array before dispatch,
the Core ML framework, or a per-backend compiler. The compute-unit differences
argue against a single pre-dispatch scrub, and `.cpuAndNE` emitting Inf shows
the stack *can* return non-finite values — but neither is conclusive. Driving
the same inputs through the Swift `ModelRunner` in M4 would separate the Python
wrapper from the framework, and that is the experiment to run next.

**n = 1 base input.** The magnitude sweep scales one seeded tensor. Thresholds
like "fp16 collapses near 1e3" are properties of this input's activation
statistics and should not be quoted as general figures.

---

## Investigation status

Each investigation resolves to a finding entry, including negative results.
*"Could not reproduce at the reported magnitude on this configuration"* is a
finding, especially when the upstream report claims hardware dependence.

| # | Question | Status |
|---|---|---|
| 1 | Does #2625's fp16 L1 drift (0.0394 vs. 0.0165) reproduce on M4 Pro / macOS 26.5.2? | **Characterized** — direction yes, magnitude no; two confounds eliminated (F-006, F-009, F-010) |
| 2 | What fraction of inputs disagree between `.cpuOnly` and `.cpuAndNeuralEngine`, and by how much? | **Complete** — 3/200 top-1 flips on synthetic inputs, with the full distribution in F-008 |
| 3 | Where does `MLModelComparator` op localization stabilize under an `atol` sweep? | **Complete** — two plateaus per arm; both arms agree on `conv:input_3` over `atol` 1e-3–1e-2, and the tolerance is shown to control search depth (F-012) |
| 4 | Run-to-run variance on a fixed input, per compute unit | **Complete** — zero maximum spread across 20 repeats/unit (F-008) |
| 5 | What ANE residency does MobileNetV3-Large actually achieve? | **Pending** |
| 6 | Any NaN/Inf-producing ops via `MLModelValidator`? | **Complete** — one Inf-producing op, only on `fp16` + `.cpuAndNE` at extreme input scale; the NaN scan never fired, and Core ML sanitizes NaN/Inf input that PyTorch propagates (F-013) |
