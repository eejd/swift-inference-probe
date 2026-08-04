# Test Strategy — InferenceProbe

**Status:** living draft, updated 3 Aug 2026. Thresholds marked ⟨measure⟩ are
deliberately unset; they require additional measurements rather than a chosen
round number. See §4.3.

---

## 1. What is under test

**The inference runtime, not the model.** MobileNetV3's ImageNet accuracy is
torchvision's problem. Whether Core ML executes MobileNetV3 faithfully, on four
compute units, at two precisions, without leaking, hanging, or silently
returning garbage, is ours.

That distinction drives everything below. It is why the primary assertion is
*agreement between execution paths* rather than *accuracy against labels*, and
why a test can fail while top-1 accuracy is unchanged — and should.

Explicitly **not** validated: model quality as a research question, training and
data pipelines, non-Apple backends, adversarial robustness, UI correctness. A
green run means the runtime behaved; it does not mean the model is good.

---

## 2. The oracle problem

Conventional software testing assumes an oracle: a known-correct expected value
you assert against. ML inference frequently has none.

- Different compute units are **not bit-identical to one another**. On this M4
  Pro each unit was bit-deterministic across repeated runs (F-008), but CPU,
  GPU, and Neural Engine outputs differ systematically because their numerical
  execution paths differ.
- coremltools converts to **fp16 by default**. Divergence from the fp32 PyTorch
  reference is *expected behaviour*, not a bug, so `assertEqual` is not merely
  brittle — it is asserting something false.
- For generative and autoregressive models there is no single correct output at
  all.

So `assertEqual` is replaced with three mechanisms, each answering a question the
others cannot.

### 2.1 Differential testing — the runtime is its own oracle

Run the same model on `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, and
`.all`. They must agree within tolerance. No ground truth is required: the
question "which is right?" is replaced by "do they agree?", and disagreement is
a defect regardless of which path is wrong.

`.cpuOnly` is the anchor (SPEC §2.3), so every comparison has a single stable
referent and failures point somewhere specific. It is privileged as the
simplest dispatch path least likely to change under a runtime update, not
because it is uniquely deterministic: F-008 found all four paths deterministic
on this configuration.

This is the highest-value mechanism in the suite. It requires no labels, no
reference framework, and it targets a real, hard bug class — the one a runtime
validation team exists to catch.

### 2.2 Golden tests with an explicit tolerance policy

Snapshot the reference output; assert future runs stay within a documented
tolerance. The assertion is **on drift, not on bit-equality**.

Goldens are environment-stamped (chip, OS, package versions). A golden blessed
on one chip is not automatically valid on another — that is the whole point of
§4.4.

### 2.3 Metamorphic relations

Properties that must hold *between related inputs*, requiring no ground truth
for either. If brightness +5% flips top-1, something is wrong, and we never
needed to know the true label.

---

## 3. The debate this project takes a position on

There is an open argument in [coremltools #2625](https://github.com/apple/coremltools/issues/2625):
converting MobileNetV3-Large to fp16 Core ML produces **L1 drift ≈ 0.0394 vs.
0.0165** for the equivalent PyTorch fp32→fp16 step — roughly 2.4× larger. A
commenter objects that end-to-end accuracy barely moves: **0.7470 (fp16) vs.
0.7474 (fp32)** across 80,040 ImageNet variants. So is the drift a defect at all?

**Our position: assert on both, at different gates, because they answer
different questions.**

**Tensor drift is a change detector.** It is cheap, sensitive, and early. It does
not tell you users are harmed; it tells you *something moved*. Its job is to
catch a regression on the PR that introduced it, while the diff is still small
enough to read.

**Task metrics are the ship gate.** They tell you whether the movement matters.
But they are expensive, statistically noisy at small N, and lagging.

The failure mode of gating **only on task metrics**: a genuine runtime defect
that happens not to move top-1 on ImageNet ships. It then surfaces on a different
model, a different input distribution, or a different chip. *Benchmark accuracy is
not a safety property of a runtime* — it is one sample of one model's behaviour
on one dataset.

The failure mode of gating **only on tensor drift**: you page someone over a
cosmetic 1e-3 and the suite gets ignored.

So: **tensor drift blocks the PR** (fast, sensitive, names the layer). **Task
metrics block the release** (slow, nightly). Critically, a tensor-drift failure
that task metrics later clear is **not a false positive** — it is a finding, and
it gets triaged to one of three outcomes: explained, fixed, or accepted into the
baseline *with a recorded reason*. Silently widening a threshold to make a test
green is the failure mode this process exists to prevent.

---

## 4. Tolerance policy

### 4.1 A single global tolerance is wrong

Different failure modes are visible to different statistics, and a policy that
uses one number for everything will be simultaneously too loose to catch real
defects and too tight to survive normal fp16 noise.

| Criterion | Detects | Blind to |
|---|---|---|
| `absolute(atol:)` | large deviations on small-magnitude outputs | proportional error on large values |
| `relative(rtol:)` | proportional error | catastrophic cancellation near zero |
| `mixed(atol:rtol:)` | numpy `allclose` semantics; good general default | systematic small bias across all elements |
| `cosineSimilarity(min:)` | direction change in the logit vector | uniform scaling of all logits |
| `topKAgreement(k:minOverlap:)` | user-visible ranking change | large numeric drift that preserves order |
| `klDivergence(maxNats:)` | confidence-distribution change | changes below softmax resolution |
| `l1Mean(max:)` | average magnitude drift; comparable to #2625 | localized single-element blowups |

The design conclusion: **each test names the criterion matching the failure it
defends against, and states why.** `TolerancePolicy.rationale` is a required,
non-optional `String` (SPEC §4.2) so this cannot be skipped — the type system
enforces the documentation, not code review.

### 4.2 Which test uses which

| Test | Criterion | Reasoning |
|---|---|---|
| fp32 Core ML vs. PyTorch fp32 | `mixed(atol:rtol:)`, tight | same precision, same math; near-equality is a legitimate expectation, and drift here means a conversion bug |
| fp16 Core ML vs. PyTorch fp32 | `l1Mean` + `topKAgreement(k:5)` | fp16 drift is expected; `l1Mean` is directly comparable to #2625's reported figure, `topK` asks whether it is user-visible |
| `.cpuOnly` vs. `.cpuAndNeuralEngine` | `mixed` + `cosineSimilarity` | same weights, same precision, different silicon — disagreement beyond fp noise is a runtime defect, and cosine catches direction change independent of scale |
| Metamorphic (brightness, flip, re-encode) | `labelInvariant` | the property under test is user-visible stability, not numeric identity |
| Metamorphic (batch-1 vs batch-N) | `mixed`, very tight | batching must not change per-item math *at all*; this is an exact invariant and deserves a near-exact assertion |

Note the last row. Batching invariance is the one place where a near-exact
assertion is correct, because there is no numerical reason for per-item outputs
to differ. It is also a real bug class that finds real bugs.

### 4.3 Thresholds are measured, not chosen

Thresholds come from the measurement that matches the failure mode. For
cross-unit comparisons, first run a fixed input N times per unit to establish
the noise floor, then measure the distribution across many inputs. The F-008
noise-floor study is complete and found zero spread; remaining ⟨measure⟩ entries
need either representative real inputs or the unfinished compute-plan study.

| Threshold | Value | Source |
|---|---|---|
| **Run-to-run noise floor** | **0.0** — all four units bit-identical | F-008, 20 repeats/unit |
| fp32 parity mixed tolerance | ⟨measure⟩ — F-009's largest per-input L1 mean was 3.54e-6, but it did not collect elementwise max at scale | F-009, n=200 across two distributions; needs real inputs and the matching metric |
| fp16 `l1Mean` max | ⟨measure⟩ — synthetic mean 0.0117–0.0132, max 0.0203 | F-009, n=200 across two distributions; needs real inputs |
| ANE-vs-CPU `l1Mean` max | **0.030** | F-008 p95 0.0218, max 0.0242 → ~25% headroom |
| GPU-vs-CPU `l1Mean` max | **0.020** | F-008 p95 0.0124, max 0.0149 → ~35% headroom |
| ANE-vs-CPU `cosineSimilarity` min | **0.9990** | F-008 min observed 0.999574 |
| ANE residency floor, **fp16** | **100%** — assert equality, not a floor | F-014, all 182 scheduled ops and 100% of cost |
| ANE residency floor, **fp32** | **n/a** — 0 ops can reach the ANE | F-014; `.cpuAndNE` compiles to a pure-CPU plan |
| Perf regression trigger | 10% over baseline | default; revisit against measured p95 jitter |

**The noise floor being exactly zero is what licenses tight thresholds here.**
Normally a gate must clear measurement jitter before it can detect anything, and
that jitter sets a floor on sensitivity. With a deterministic runtime there is no
such floor — the only headroom needed is for legitimate variation across inputs,
which is why the values above sit just above the observed max rather than
multiples of it.

"I set the gate from measured variance rather than picking a number" is the
distinction between a threshold that holds and one that gets widened the first
time it fires.

### 4.4 Tolerance is a function of the host chip

#2625 reports an M1 Air showing high numerical error on ops where an M1 Max is
fine. If hardware-dependent divergence is real, one global tolerance is wrong by
construction, so `TolerancePolicy` carries per-chip overrides and every result is
stamped with `HostEnvironment`.

This project runs on **M4 Pro / macOS 26.5.2** — neither chip from the issue —
so our numbers are a third data point regardless of whether the repro succeeds.

### 4.5 Op localization is itself tolerance-dependent

The #2625 reporter found `MLModelComparator` flagging ordinary `linear` and
`conv` layers *depending on the `atol` passed*. Localization is not a fixed
answer; it is a function of a parameter.

So `localize.py` sweeps `atol` across several decades and reports the flagged-op
count at each. Expected shape: everything flagged at very low `atol`, nothing at
very high, and a **knee** in between where the answer is stable. That knee is the
defensible operating point, and the sweep curve is the evidence for it.

---

## 5. Test categories, by risk

Risk here means *consequence of the defect reaching a user*, which is what
decides whether a test blocks a merge.

### 5.1 Tier 1 — ships a broken feature (blocks PR)

- **NaN / Inf in output.** Unconditional failure, no tolerance debate. Detected
  by `MLModelValidator.find_failing_ops_with_nan_output()`. **This is a
  sufficient condition for failure and a weak necessary one** — F-013 found the
  NaN scan never firing across 56 configurations, and the Inf scan firing only
  on `fp16` + `.cpuAndNE` at extreme input scale, because Core ML sanitizes
  malformed input and saturates rather than overflowing. Pair it with the next
  item; a green NaN/Inf scan is not evidence of a healthy model.
- **Finite but implausible output.** The failure mode that actually occurs here:
  a confident, finite, wrong answer. Gated on output magnitude against the fp32
  reference and on `.cpuOnly`-vs-`.cpuAndNE` agreement, both of which flagged
  every broken case in F-013 including the ones the validator missed.
- **Top-1 flip on a metamorphic invariant.** The user sees a wrong answer.
- **Compute-unit disagreement beyond tolerance.** One of the four paths is
  wrong; whichever it is, some fraction of devices gets bad output.
- **Crash / hang on malformed input.** Wrong shape, empty, truncated.
- **Thread-safety failure** under concurrent `prediction()`. Non-deterministic
  and near-impossible to debug in the field.

### 5.2 Tier 2 — degrades quality (blocks release, not PR)

- fp16 `l1Mean` drift beyond baseline while top-k holds.
- Latency p95 regression >10%.
- Peak memory regression.
- ANE residency drop (see §5.3).

### 5.3 The compute-plan assertion

The most distinctive test in the suite, and worth calling out separately.

Core ML dispatches each op to the fastest device that supports it — Neural Engine
for most matmuls, convs, and attention; GPU for what ANE cannot do; CPU for the
rest. `MLComputePlan` exposes the anticipated assignment per op *before you run
anything*.

So: **assert that ≥100% of ops are Neural-Engine resident, and fail if a change
pushes ops back to CPU.**

F-014 measured that figure and it came back at the ceiling: **all 182 scheduled
ops, 100% of estimated cost, on the ANE** — but only at fp16. At fp32 *no* op
can reach the ANE, `.cpuAndNE` compiles to a pure-CPU plan, and `.all` picks the
GPU. Two consequences for how this assertion is written:

- Because residency is 100% rather than some partial figure, the assertion is an
  **equality, not a threshold**. Any op moving off the ANE is a regression, and
  there is no headroom to argue about.
- **`.cpuAndNE` is not an ANE test unless the model is fp16.** A parity suite
  that runs an fp32 model under `.cpuAndNE` is silently testing the CPU path
  twice.
- **Assert on device assignment, never on the cost estimate.** F-014 found the
  assignment bit-stable across plan loads while the per-op cost weights wobble
  in their last ulp. A golden keyed on the raw plan would flake with nothing
  changed.

This is a performance regression test **on the compilation result rather than the
wall clock**. Two consequences:

1. It catches the regression *before* it manifests as latency.
2. It is **stable on a noisy CI machine**, where timing is not. Wall-clock
   assertions on shared runners are the classic source of flaky perf tests; this
   one has no timing component at all.

### 5.4 Tier 3 — cosmetic

Sub-threshold numeric drift with no ranking change; load-time variance within
noise. Recorded in trends, asserted on loosely or not at all.

---

## 6. What runs when, and the budget that decided it

GitHub Actions bills macOS at a **10× multiplier**, so the 2,000-minute free tier
is ~200 real macOS minutes per month. That constraint, not preference, sets the
split.

### PR gate — target <5 minutes

Build · Swift Testing unit suite · parity on MobileNetV3 under `.cpuOnly` and
`.all` only · NaN/Inf validation · compute-plan assertion (cheap: no inference
required) · `xcresulttool` posts a readable report to the PR.

At ~5 min/run, ~40 PR runs/month fit the free tier. Everything else is nightly by
necessity.

### Nightly — full matrix

3 models × 4 compute units · fp32 and fp16 · full metamorphic set · robustness
and fault injection · performance suite against committed baselines · `atol`
sweep · `xcresultparser` → JSON → latency trend published to Pages.

Designing the gate around an explicit time budget, and moving the compute-unit
matrix to nightly to hold it, is the decision — not an implementation detail.

---

## 7. Flake budget and quarantine

Flaky tests are more corrosive than missing ones: a suite people have learned to
re-run is a suite that no longer gates anything.

- **Budget: <1% failure rate per test over a rolling 20 runs.** Above that, a
  test is quarantined rather than deleted or ignored.
- **Retry-on-failure** via `xcodebuild -retry-tests-on-failure -test-iterations N`
  (verified available as a CLI flag, so this needs no test plan), and the **retry
  rate is logged as a metric** — a test that passes on retry is passing, but the
  retry itself is data, and a rising retry rate is an early warning.
- **Flake hunting** uses `-run-tests-until-failure` on a suspect test, rather
  than re-running the whole suite and hoping.
- **Quarantine** uses `withKnownIssue` (Swift Testing) / `XCTExpectFailure`
  (XCTest). The test **keeps running** and reports without failing the build — so
  when it starts passing consistently, we find out. Disabling a test hides it;
  quarantining it keeps the signal.
- Every quarantined test needs an owner and a linked issue. Quarantine is a
  waiting room, not a graveyard.

---

## 8. Known gaps

Stated plainly, because a strategy document that claims full coverage is not
credible.

1. **One model at first.** MobileNetV3 only; DistilBERT and Whisper follow.
   Generality is a claim the harness has not yet earned.
2. **Mac only.** No on-device iPhone validation yet. coremltools' Remote-Device
   utility (`Device.get_connected_devices()` → `prepare_for_model_debugging()`)
   would extend every parity and perf test to real silicon over `devicectl`, and
   needs an Apple Developer Team ID. This is the largest single gap, given the
   word "on-device."
3. **No `.xctestplan` files** until the Xcode project exists. Measured 1 Aug:
   `xcodebuild` CLI flags cover selection, retry, and parallelization, so the
   functional loss is small. The real loss is that configuration lives in
   workflow YAML instead of a committed, code-reviewable artifact (SPEC §2.1).
4. **Single chip.** All thresholds are M4 Pro-derived. The per-chip override
   mechanism exists but has one data point, so it is untested as a mechanism.
5. **No performance baselines committed yet**, so the 10% regression gate is
   inactive until Day 5+.
6. **Experimental API surface.** The comparator/validator/benchmarker utilities
   are `coremltools.models.ml_program.experimental` and may change between
   releases; the pin in `requirements.txt` is load-bearing.
7. **Distributed / Private Cloud Compute inference is entirely out of scope.**
   The harness validates on-device only.
