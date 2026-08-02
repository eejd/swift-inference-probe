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
| Swift | 6.0 (swiftlang-6.0.0.9.10) via CommandLineTools |
| Python | 3.12 (dedicated conda env `inferenceprobe`) |
| torch | *pending — Day 1* |
| torchvision | *pending — Day 1* |
| coremltools | *pending — Day 1* |

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
non-comparable to local ones. Fix is `sudo xcode-select -s`; `DEVELOPER_DIR`
works without sudo. **Resolution: pending.**

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
