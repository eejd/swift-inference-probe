# InferenceProbe — Specification

**Status:** living draft, updated 3 Aug 2026. M0–M2 are complete, M3 is
partial, and the component contracts below remain the target for M4.

A test harness that validates Core ML inference on Apple silicon for numerical
correctness across compute units, robustness, and performance regression, with a
CI gate on every PR.

Companion documents: [`TEST_STRATEGY.md`](TEST_STRATEGY.md) (what we validate and
why), [`ROADMAP.md`](ROADMAP.md) (execution order), [`FINDINGS.md`](FINDINGS.md) (what it
caught).

---

## 1. Scope

### In scope

The **inference runtime** is the system under test. We validate that a model,
once exported, produces acceptable outputs across compute units, precisions, and
input perturbations, and that it does so within a performance envelope.

1. **Numerical parity** — Core ML output vs. the PyTorch reference, under an
   explicit tolerance policy.
2. **Compute-unit differential** — `.cpuAndGPU` / `.cpuAndNeuralEngine` /
   `.all` agreement against the `.cpuOnly` anchor.
3. **Compute-plan assertions** — op-to-device assignment as a first-class
   testable property, not just a performance side effect.
4. **Metamorphic relations** — properties that must hold between related inputs,
   requiring no ground truth.
5. **Robustness** — malformed input, NaN/Inf, concurrency, load/unload leaks,
   cancellation.
6. **Performance regression** — latency percentiles, cold vs. warm load, peak
   memory, against committed baselines.

### Explicitly out of scope

Stated here so a reader knows what a green run does *not* mean.

- **Model quality as a research question.** Whether MobileNetV3 is a good
  classifier is not our problem. Whether the runtime faithfully executes it is.
- **Training, fine-tuning, data pipelines.** We consume an exported artifact.
- **Non-Apple backends.** No ONNX Runtime, no TFLite comparison.
- **Adversarial robustness / security.** Perturbation testing here is
  metamorphic, not adversarial.
- **UI correctness.** The demo app exists only to give XCUITest and
  `XCTApplicationLaunchMetric` a surface. It is not a product.

---

## 2. Key decisions

Each of these is a decision with a live alternative, recorded so it can be
revisited rather than rediscovered.

### 2.1 SwiftPM is the source of truth; `xcodebuild` drives it in CI

`Package.swift` defines the library and every test target that does not require
an app host. Xcode opens a package natively, so the IDE — test navigator,
debugger, Instruments — is fully available without an `.xcodeproj`.

**CI uses `xcodebuild`, not `swift test`.** This was reversed on 1 Aug after
measurement: `swift test` produces no `.xcresult` bundle, and `.xcresult` is what
the entire reporting layer consumes (`xcresulttool` PR comments, `xcresultparser`
→ trend dashboard, `flake-radar` later). `xcodebuild` drives a bare package
happily, so the bundle costs nothing structurally.

```bash
# scheme is auto-named <PackageName>-Package — NOT <PackageName>
xcodebuild test -scheme InferenceProbe-Package -destination 'platform=macOS' \
  -resultBundlePath result.xcresult
```

Verified on Xcode 26.6 / macOS 26.5.2 / M4 Pro. See [`FINDINGS.md`](FINDINGS.md)
F-003 for the full evidence.

**What we lose without an `.xcodeproj`, and what substitutes:**

| Test-plan feature | Substitute | Status |
|---|---|---|
| PR/nightly test selection | `-only-testing:` / `-skip-testing:` + Swift Testing tags | ✅ equivalent |
| Retry-on-failure (flake budget) | `-retry-tests-on-failure`, `-test-iterations N` | ✅ equivalent |
| Until-failure flake hunting | `-run-tests-until-failure` | ✅ equivalent |
| Parallelization | `-parallel-testing-enabled`, `-maximum-parallel-testing-workers` | ✅ equivalent |
| **Performance baselines** (`.xcbaseline`) | commit our own baseline JSON; compare from `xcresulttool get test-results metrics` | ⚠️ hand-rolled — see §4.8 |
| Declarative, reviewable config | none — config lives in workflow YAML | ❌ genuine loss |

Only the last two are real losses. The configuration one is worth naming
honestly: a `.xctestplan` is a committed artifact that shows up in code review,
whereas CLI flags hide in YAML.

**No `.xcodeproj` may sit in the package root.** This is a constraint, not a
preference: `xcodebuild` resolves a project file in preference to
`Package.swift` and never falls back, so a stray project — even one with zero
targets — makes every `xcodebuild` invocation fail with an error that blames the
scheme name instead. An empty template project was removed from this repo for
exactly that reason; see [`FINDINGS.md`](FINDINGS.md) F-004.

When the demo app is added post-deadline it needs a real project with real
targets, and the CI invocation must then name that project's scheme explicitly
rather than the implicit `-Package` one.

### 2.2 Swift Testing and XCTest coexist, deliberately

Swift Testing is the 2026 default for unit tests, but it **does not cover UI
automation or performance measurement**. `XCUITest` and the `XCTMetric` family
stay in XCTest. Both frameworks live in the same target.

| Target | Framework | Why |
|---|---|---|
| `ParityTests` | Swift Testing | `#expect`, parameterization |
| `ComputeUnitTests` | Swift Testing | `@Test(arguments:)` over the compute-unit matrix |
| `MetamorphicTests` | Swift Testing | parameterization is the whole point |
| `RobustnessTests` | Swift Testing | `withKnownIssue` for quarantine |
| `PerformanceTests` | **XCTest** | `measure(metrics:)` / `XCTMetric` have no Swift Testing equivalent |
| `ProbeDemoUITests` | **XCTest** | XCUITest only |

### 2.3 `.cpuOnly` is the reference anchor

Differential tests are anchored: every other compute unit is compared *against
`.cpuOnly`*, not pairwise-all, which reduces the matrix from O(n²) to O(n) and
gives failures a single clear referent.

**Corrected 2 Aug.** This section originally justified the anchor by claiming
"GPU and Neural Engine execution is not bit-reproducible across runs." That was
inherited assumption, and it is **false on this configuration** —
[`FINDINGS.md`](FINDINGS.md) F-008 measured all four compute units as
bit-identical across repeated runs of a fixed input.

The anchor is still right, but for a different reason. `.cpuOnly` is the
*semantically privileged* path — the simplest dispatch, the one least likely to
change under a Core ML update, and the natural referent for "what should this
have computed." It is not more deterministic than the others, because none of
them are non-deterministic.

The consequence is large and load-bearing: with a zero noise floor, **every
cross-unit difference is systematic**, so thresholds can be tight and a
differential test cannot flake for numerical reasons.

### 2.4 fp16 is the default and the reason this project exists

coremltools converts fp32 PyTorch models to **fp16 by default** when targeting
Core ML. Divergence from the PyTorch reference is therefore *expected*. The
engineering question is not "is there drift" but "how much is acceptable, and how
do we detect when it stops being acceptable." Every parity test exports both
precisions so the fp16 delta is always attributable.

### 2.5 Tolerance is a function of the host chip

An open coremltools issue reports numerical divergence differing between an M1
Air and an M1 Max on the same model. If that is real, a single global tolerance
is wrong by construction. `TolerancePolicy` therefore carries per-chip overrides,
and the harness records the host chip in every result.

---

## 3. Target repo layout

This is the intended M4/M5 layout, not a claim that every file exists today.
As of 3 Aug, the Swift side contains `HostEnvironment.swift` and
`HarnessTests`; the Python side contains `export_model.py`,
`compute_unit_differential.py`, and `repro_2625.py`. Generated model packages
are local and ignored, while their manifests and measurement JSON are tracked
under `Fixtures/models*`.

```
InferenceProbe/
├── Package.swift
├── CLAUDE.md
├── Sources/
│   ├── InferenceProbe/              # the harness library
│   │   ├── ModelRunner.swift        # MLModel wrapper, compute-unit selection
│   │   ├── TolerancePolicy.swift    # abs/rel tol, cosine, top-k, KL
│   │   ├── GoldenStore.swift        # snapshot load / compare / bless
│   │   ├── MetamorphicRelation.swift
│   │   ├── Metrics.swift            # latency percentiles, ANE residency
│   │   └── HostEnvironment.swift    # chip, OS, versions — stamped on results
│   └── ProbeDemo/                   # ~80-line SwiftUI app (post-7 Aug)
├── Tests/
│   ├── ParityTests/                 # PyTorch ref vs Core ML
│   ├── ComputeUnitTests/            # cpuOnly vs GPU vs ANE vs all
│   ├── MetamorphicTests/
│   ├── RobustnessTests/
│   ├── PerformanceTests/            # XCTest
│   └── ProbeDemoUITests/            # XCUITest (post-7 Aug)
├── python/
│   ├── export_model.py              # PyTorch → Core ML via coremltools
│   ├── generate_goldens.py          # reference outputs + fixtures
│   ├── localize.py                  # MLModelComparator + atol sweep
│   ├── requirements.txt             # pinned, incl. coremltools
│   └── tests/                       # pytest mirror
├── TestPlans/                       # post-7 Aug (needs .xcodeproj)
├── Fixtures/
│   ├── inputs/                      # committed input tensors
│   └── goldens/                     # committed reference outputs + manifests
├── .github/workflows/{pr,nightly}.yml
└── docs/{SPEC,TEST_STRATEGY,PLAN,FINDINGS}.md
```

---

## 4. Component contracts

Signatures are indicative, not final; they exist to pin down responsibilities.

### 4.1 `HostEnvironment`

Every comparison result is stamped with the environment that produced it.
Without this, a golden committed on one machine is meaningless on another.

```swift
public struct HostEnvironment: Codable, Sendable, Equatable {
    public enum Chip: String, Codable, Sendable {
        case m1, m1Pro, m1Max, m1Ultra
        case m2, m2Pro, m2Max, m2Ultra
        case m3, m3Pro, m3Max
        case m4, m4Pro, m4Max
        case unknown
    }
    public let chip: Chip
    public let chipBrandString: String   // raw machdep.cpu.brand_string
    public let osVersion: String
    public let coreCount: (performance: Int, efficiency: Int)

    public static func current() -> HostEnvironment
}
```

### 4.2 `TolerancePolicy`

The comparison vocabulary. Different failure modes need different assertions;
a test picks the criterion that matches the failure it is defending against.

```swift
public struct TolerancePolicy: Sendable {
    public enum Criterion: Sendable, Codable {
        case absolute(atol: Double)
        case relative(rtol: Double)
        case mixed(atol: Double, rtol: Double)      // |a-b| <= atol + rtol*|b|
        case cosineSimilarity(min: Double)
        case topKAgreement(k: Int, minOverlap: Double)
        case klDivergence(maxNats: Double)          // over softmax
        case l1Mean(max: Double)                    // mean |a-b|, comparable to issue #2625
    }

    /// All criteria must pass. An empty array is a compile-time-legal but
    /// semantically invalid policy and traps in `init`.
    public let criteria: [Criterion]

    /// Non-optional by design. A policy cannot be constructed without stating
    /// why these thresholds and not others. This is the single most important
    /// property in the type — see TEST_STRATEGY.md §4.
    public let rationale: String

    /// Overrides applied when running on a specific chip. Empty means the
    /// base criteria apply everywhere.
    public let chipOverrides: [HostEnvironment.Chip: [Criterion]]

    public func criteria(for env: HostEnvironment) -> [Criterion]
}
```

`rationale` being a required, non-optional `String` is deliberate: it makes
"document why each test chose this threshold" a property of the type system
rather than a code-review convention.

### 4.3 `ComparisonResult`

A failure must name the offending layer, not merely report that outputs differ.

```swift
public struct ComparisonResult: Sendable {
    public struct CriterionOutcome: Sendable {
        public let criterion: TolerancePolicy.Criterion
        public let measured: Double
        public let threshold: Double
        public let passed: Bool
    }
    public let outcomes: [CriterionOutcome]
    public let environment: HostEnvironment
    public let worstIndex: [Int]?        // tensor coordinate of max deviation
    public let suspectOps: [String]      // populated by the Python localizer
    public var passed: Bool { outcomes.allSatisfy(\.passed) }

    /// Renders the failure message: criterion, measured vs. threshold,
    /// worst coordinate, suspect ops, host chip.
    public var failureDescription: String { get }
}
```

### 4.4 `ModelRunner`

```swift
public struct ModelRunner: Sendable {
    public enum Precision: String, Sendable { case fp32, fp16 }

    public init(modelURL: URL, computeUnits: MLComputeUnits) throws

    public func predict(_ input: MLFeatureProvider) throws -> MLFeatureProvider

    /// Cold = compile from .mlpackage; warm = load cached .mlmodelc.
    /// These are distinct and commonly-conflated costs.
    public func measureLoad(cold: Bool) throws -> Duration

    /// Op-to-device assignment from MLComputePlan.
    public func computePlan() async throws -> ComputePlanSummary
}
```

### 4.5 `ComputePlanSummary`

The basis for compute-plan assertions — a regression test on the *compilation
result* rather than the wall clock. See TEST_STRATEGY.md §5.3.

```swift
public struct ComputePlanSummary: Sendable, Codable {
    public struct OpAssignment: Sendable, Codable {
        public let opName: String
        public let opType: String
        public let preferredDevice: String     // ANE | GPU | CPU
        public let supportedDevices: [String]
    }
    public let assignments: [OpAssignment]
    public var totalOps: Int { assignments.count }
    public func residency(on device: String) -> Double   // fraction in [0,1]
}
```

### 4.6 `GoldenStore`

```swift
public protocol GoldenStore: Sendable {
    func load(_ id: GoldenID) throws -> Golden
    func compare(_ actual: MLMultiArray,
                 to id: GoldenID,
                 using policy: TolerancePolicy) throws -> ComparisonResult

    /// Regenerating a golden requires a stated reason, recorded in the
    /// manifest alongside the environment that produced it. Blessing is a
    /// deliberate act with an audit trail, not a `--fix` flag.
    func bless(_ actual: MLMultiArray, as id: GoldenID, reason: String) throws
}
```

Golden manifest format (`Fixtures/goldens/<id>.json`) — the tensor itself is a
sibling `.npy`:

```json
{
  "id": "mobilenetv3_large.fp32.cpuOnly.imagenet_sample_000",
  "producedBy": { "chip": "m4Pro", "osVersion": "26.5.2",
                  "coremltools": "TBD", "torch": "TBD" },
  "blessedAt": "2026-08-01T00:00:00Z",
  "reason": "initial baseline",
  "shape": [1, 1000],
  "dtype": "float32",
  "tensor": "mobilenetv3_large.fp32.cpuOnly.imagenet_sample_000.npy"
}
```

### 4.7 `MetamorphicRelation`

```swift
public protocol MetamorphicRelation: Sendable {
    associatedtype Input
    var name: String { get }
    var rationale: String { get }
    func transform(_ input: Input) throws -> Input
    var expectation: Expectation { get }
}

public enum Expectation: Sendable {
    case outputInvariant(TolerancePolicy)   // logits must match within tolerance
    case labelInvariant                     // top-1 must not flip
    case topKStable(k: Int, minOverlap: Double)
}
```

### 4.8 `Metrics`

`XCTest`'s `measure()` reports **mean and standard deviation, not percentiles**.
Latency distributions for inference are right-skewed, so the mean understates
tail behaviour and stddev is not a useful summary of it. `Metrics` layers a
percentile harness on top.

**Measured 1 Aug (F-003), and it changes the design.** `measure(metrics:)` runs
fine under `swift test` — no Xcode project needed — and
`xcrun xcresulttool get test-results metrics --path <bundle>` returns **raw
per-iteration measurement arrays**, not just aggregates. So percentiles are
computed in post-processing from data XCTest already collected, rather than
hand-rolled in a parallel timing harness.

The real gap is elsewhere: without an `.xcodeproj` there is **no committed
baseline**, so `measure()` reports `baselineName: ""` and the `maxPercentRegression: 10%`
gate is inert — it measures but cannot fail. The fix is to commit our own
baseline JSON and compare in CI, which also lets the gate assert on **p95 rather
than mean** — something Xcode baselines cannot do.

```swift
public struct LatencyStats: Sendable, Codable {
    public let n: Int
    public let p50, p90, p95, p99: Duration
    public let mean, stddev: Duration
    public let min, max: Duration
    public static func over(iterations: Int,
                            warmup: Int,
                            _ body: () throws -> Void) rethrows -> LatencyStats
}
```

---

## 5. Python side

The Swift harness asserts; Python produces references and does op-level
localization, because Apple's debugging utilities are Python-only.

| Script | Responsibility | Status |
|---|---|---|
| `export_model.py` | torchvision → Core ML at fp32 and fp16; emits a version manifest | Implemented |
| `compute_unit_differential.py` | Repeat-run noise floor and all compute units against `.cpuOnly` | Implemented |
| `repro_2625.py` | fp16 drift across controlled synthetic input distributions | Implemented |
| `generate_goldens.py` | PyTorch reference outputs + committed input fixtures | Planned, M4 |
| `localize.py` | `MLModelComparator` op localization, swept across `atol` | Pending, M3 |
| `validate.py` | `MLModelValidator.find_failing_ops_with_nan_output()` / `_infinite_output()` | Pending, M3 |
| `bench.py` | `MLModelBenchmarker.benchmark_load()` / `.benchmark_predict(warmup=True)` / `.benchmark_operation_execution()` | Planned |

**Version pinning is mandatory.** `MLModelInspector`, `MLModelValidator`,
`MLModelComparator`, and `MLModelBenchmarker` live under
`coremltools.models.ml_program.experimental` and are flagged experimental by
Apple. `requirements.txt` pins exact versions and `export_model.py` writes the
resolved versions into every artifact manifest.

**Interpreter:** the machine's default `python3` is Homebrew **3.14.2**, which is
ahead of torch/coremltools support. The project uses a dedicated conda env at
Python **3.12**. See [`ROADMAP.md`](ROADMAP.md) §Environment.

---

## 6. Open questions

Tracked here rather than guessed at; each is resolved by a measurement in
[`ROADMAP.md`](ROADMAP.md).

1. ~~**Does coremltools #2625 reproduce on M4 Pro / macOS 26.5.2?**~~
   **Characterized 3 Aug:** the direction reproduces but the reported magnitude
   does not. Input distribution and torch version are eliminated; hardware/OS
   remains the leading explanation but is not proven. See F-006, F-009, F-010.
2. ~~**What is the run-to-run variance on a fixed input, per compute unit?**~~
   **Resolved 2 Aug:** zero maximum spread across 20 repeats for all four units
   on this configuration. See F-008.
3. **Where is op localization stable under `atol`?** The issue reporter found
   `MLModelComparator` blames ordinary `linear`/`conv` layers depending on
   `atol`. A sweep should show a knee. *(Day 3)*
4. **What ANE residency does MobileNetV3 actually achieve?** Sets the threshold
   for the compute-plan assertion. *(Day 3)*
5. ~~**Does `measure(metrics:)` work under `swift test`, or does it require an
   Xcode scheme?**~~ **Resolved 1 Aug — yes, it works.** But baselines do not,
   so the regression gate needs our own baseline file. See §4.8 and FINDINGS
   F-003.
