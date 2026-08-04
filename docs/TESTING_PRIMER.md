# Testing systems that have no right answer

A primer on validating ML inference, written against this repository's actual
findings rather than in the abstract. Every example below is something the
harness measured; the finding IDs point at [`FINDINGS.md`](FINDINGS.md).

For *what* this project validates, see [`TEST_STRATEGY.md`](TEST_STRATEGY.md).
This document is about *why testing ML systems needs different tools at all*.

---

## 1. The idea everything else hangs from

Ordinary software testing rests on an assumption so basic it is rarely named:
**you know what the answer should be.**

```
assertEqual(add(2, 2), 4)
```

That `4` is the **test oracle** — the source of truth the assertion compares
against. Every `assertEqual` ever written borrowed one from somewhere: a
specification, a hand-worked example, someone's head.

Now write a test for a model that classifies an image. What is the oracle?

The obvious answer, "the correct label," tests the *model's accuracy* — a
research question the model already answered as well as it is going to. It says
nothing about whether the **runtime** — the component that loads the weights and
executes the math on a particular chip — did its job. The runtime is what we are
testing.

Exact comparison is not merely brittle here, it is wrong. coremltools converts
models to **fp16 by default**, so Core ML output *always* differs from the
PyTorch fp32 original. An `assertEqual` would be asserting something false.

This is the **test oracle problem**. It is the single idea that separates
testing an ML system from testing a calculator.

---

## 2. Three ways to test without an oracle

### 2.1 Differential testing — make the system its own oracle

The move is a reframe. Instead of asking *"which output is correct?"* —
unanswerable — ask *"do these paths agree?"*

Core ML can execute one model four ways: CPU only, CPU+GPU, CPU+Neural Engine,
or unconstrained. Same file, same weights, same input. **They must agree.** If
they do not, one of them is wrong, and it is a defect regardless of which.

No labels, no reference framework, no ground truth. This is the highest-value
mechanism in this repository, and `python/compute_unit_differential.py` is the
implementation.

### 2.2 Golden tests with a tolerance policy

Record a reference output once; compare future runs against it **within a stated
tolerance** rather than exactly. The assertion is on *drift*, not equality.

"Stated" carries the weight. Section 4.2 explains why.

### 2.3 Metamorphic relations

Test **relations between inputs** instead of outputs directly. Brighten an image
by 5% and the predicted class must not change. You never needed the true label —
only a property that must hold between a pair of inputs.

---

## 3. The method, as practised here

### Step 1 — Gate the environment before building anything

The first work in this project was an inventory: Xcode, Swift, Python, disk. It
immediately found two blockers — `xcode-select` pointing at the wrong toolchain
(F-001) and a default Python ahead of PyTorch support (F-002).

The principle: **a spike is cheap, a surprise is expensive.** Converting unknowns
into knowns is worth most at the start, when there is still time to react.

### Step 2 — Write the strategy before the tests

[`TEST_STRATEGY.md`](TEST_STRATEGY.md) states what is validated, what is
explicitly *not*, and the reasoning behind every threshold. Without it, a suite
is a pile of assertions nobody can evaluate. When someone asks "is 0.02 the right
tolerance?", the document answers; otherwise the honest answer is "I guessed."

### Step 3 — Build a control into the experiment

The model is exported at **fp32 and fp16**, and the fp32 export is the reason the
fp16 number means anything.

Export only fp16, observe drift, and you cannot tell whether it came from the
precision change or from a broken conversion. The fp32 path came back at
L1 = 0.000002 — essentially exact (F-006). That is a **control**: it proves the
conversion is faithful and isolates the drift to precision alone.

The same logic drives the PyTorch fp32→fp16 baseline in `export_model.py`. Some
drift is simply what fp16 costs; only the *excess* over that baseline is
attributable to Core ML.

### Step 4 — When something breaks, bisect it

The fp16 export died with exit code 139 — SIGSEGV, no Python traceback. The
tempting conclusion, "fp16 conversion is broken," was wrong.

Varying one thing at a time (F-005):

| Sequence in one process | Result |
|---|---|
| `convert(fp16)` alone | ✅ |
| `convert(fp32)` → `convert(fp16)` | ✅ |
| `convert` → `save` → `convert` → `save` | ✅ |
| `convert` → `save` → `load` → **`predict`** → `convert` | ❌ **segfault** |

The defect is that `ct.convert()` crashes if a `predict()` has already run in the
same process — an *interaction between two calls*, not a property of either.

That table is what a defect report should look like. Compare it to "fp16 export
crashes." One of those a maintainer can act on; the other starts a long
conversation.

**Bisection localises; it does not explain.** The table says *where* the bug
lives, which was enough to work around it — but not *why* a `predict()` should
poison a later `convert()`. The explanation arrived from a second environment
built for an unrelated reason (Step 7). The identical sequence on torch 2.7.0
aborts with a message rather than dying silently:

| Environment | Exit | Signal |
|---|---|---|
| torch 2.13.0 | 139 | SIGSEGV — raw memory fault, no message |
| torch 2.7.0 | 134 | SIGABRT, with a diagnostic |

```
Fatal Python error: PyEval_SaveThread: the function must be called with the GIL
held … but the GIL is released
```

The root cause is **GIL mishandling in coremltools' native extension**, not
memory corruption: something releases Python's global lock and then re-enters a
path that assumes it still holds it. That is exactly why the bug is an
interaction — `predict()` leaves the interpreter's thread state in a condition
that `convert()`'s implicit model-load path mishandles.

Two things generalise. **A crash that talks is worth more than a crash that does
not**, so reproducing a native fault in a second environment is often cheaper
than debugging it in the first. And the fix the bisection produced —
`skip_model_load=True` — was **correct independently of the bug**: the implicit
load it removes uses `ComputeUnit.ALL`, which is not the pinned `.cpuOnly`
reference anchor [`SPEC.md`](SPEC.md) §2.3 requires. Working around the defect
corrected a real design error.

### Step 5 — Measure the noise before setting any threshold

This step is the one most often skipped, and here it changed a conclusion.

Before asking *"how much do the compute units disagree?"*, ask **"how much does
one compute unit disagree with itself?"** Same input, same unit, 20 times.

The answer was **exactly zero** — all four compute units bit-identical run to run
(F-008).

Why this matters: had the Neural Engine wobbled by ±0.02 between runs, the 0.019
cross-unit disagreement would have been *indistinguishable from noise* — a number
with no meaning. Because the floor is zero, every difference measured is real and
systematic.

It also falsified an assumption written into [`SPEC.md`](SPEC.md) §2.3 — that GPU
and ANE execution is not bit-reproducible. That is conventional wisdom, and on
this hardware it is false. The spec was corrected and the reason recorded.
**Measured beats assumed, including when the assumption was your own.**

### Step 6 — Run the differential at scale

200 inputs, each compute unit against the `.cpuOnly` anchor (F-008):

| Unit | L1 mean | cosine min | top-1 flips |
|---|---|---|---|
| `.cpuAndGPU` | 0.010203 | 0.999826 | 2 / 200 |
| `.cpuAndNE` | 0.018858 | 0.999574 | **3 / 200 (1.5%)** |

**1.5% of inputs produce a different predicted class on the Neural Engine than on
the CPU.** Same file, same weights, same input.

Two structural observations came out of the same experiment (F-007, at n=1, and
confirmed by the table above), and both changed where the project points its
effort.

**The compute path moves the output more than the precision change does.** The
ANE-vs-CPU disagreement of 0.0189 is larger than the entire fp16 conversion
drift of 0.0113 (F-006). fp16 is the famous problem; on this model it is the
smaller term. **Test the effect you measured, not the one you expected** — which
is why the differential suite, not the fp16 suite, is where the value is.

**`.all` is bit-identical to `.cpuAndNE`**, so Core ML dispatches this model to
the Neural Engine when given a free choice. That does not make testing both
redundant: **the equality is itself the assertion.** Some invariants are worth
asserting precisely because you expect them to hold — the day `.all` stops
matching `.cpuAndNE`, dispatch has changed underneath you, and nothing else in
the suite would have said so.

### Step 7 — Name the alternative explanations, then close them one at a time

The first drift measurement came in at **1.78×** where the published issue
reported **2.39×** (F-006). The honest reading of that is neither "reproduced"
nor "failed to reproduce". It is: *three things differ between this machine and
that one, and this measurement separates none of them.*

| Confound | Status |
|---|---|
| **Input distribution** — synthetic noise vs. real photographs | closed, F-009 |
| **torch version** — 2.13.0 vs 2.7.0 | closed, F-010 |
| **Chip / OS** — M4 Pro / macOS 26.5.2 vs M1 / 26.1 | open, not testable here |

Writing that table is most of the work. Once the alternatives are named, each
one stops being a doubt and becomes a finite experiment.

**Input distribution (F-009).** White noise is off-distribution for a network
trained on photographs, and **L1 is not scale-free** — if logit magnitudes move,
L1 moves with them. So `python/repro_2625.py` generates a second input set with
a **1/f^1.2 power spectrum**: the spatial statistic that separates photographs
from static, where energy sits at low frequencies and neighbouring pixels
correlate.

| | white noise | natural 1/f | issue #2625 |
|---|---|---|---|
| Core ML fp16 L1 | 0.011749 | 0.013184 | 0.0394 |
| PyTorch fp16 L1 | 0.006923 | 0.008707 | 0.0165 |
| **ratio** | **1.70×** | **1.51×** | **2.39×** |
| mean logit \|max\| | **5.38** | **6.52** | — |

Read the last row before the others — §4.5 explains why. Then the result: the
ratio moved *away* from 2.39×, not toward it. Input distribution is eliminated,
and eliminated in the stronger direction, since more photograph-like inputs
would be expected to lower it further still.

**torch version (F-010).** A second environment pinned torch 2.7.0 — the version
the issue was filed against, and the newest coremltools 9.0 claims to have
tested. Every number came back **bit-identical**: the two result files differ
nowhere at all, and the manifests differ only in the version strings they
record.

**What remains.** By elimination the discrepancy is attributable to hardware or
OS — which is independently what the issue thread argues, from a different
direction, on a different chip. Two routes to one answer are worth more than
either alone. It also converts [`TEST_STRATEGY.md`](TEST_STRATEGY.md) §4.4 —
tolerance is a function of the host chip — from an inherited assumption into an
evidence-backed one.

**And elimination is not proof.** A fourth unconsidered variable — the
deployment target, the Core ML runtime shipped with one OS versus another, the
reporter's own fixture — could still account for it. That caveat belongs in the
finding rather than smoothed out of it, because a conclusion by elimination is
only ever as strong as the list of alternatives somebody thought to write down.

### Step 8 — An answer that moves with a parameter is not yet an answer

Once a test fails, the next question is *which layer*. Apple ships a tool for
exactly that: `MLModelComparator` walks the graph and reports the op responsible.

Run it once and you get a layer name, which feels like an answer. Run it across
tolerances and the name changes (F-012):

| `atol` | blamed op |
|---|---|
| 1e-6 … 1e-4 | `conv:input_3`, `conv:input_67` |
| 1e-3 … 1e-2 | `conv:input_3` |
| 1e-1 | **`linear:var_930`** |
| 1e0 | *(nothing)* |

Same model, same input, one parameter changed — and the blame moves from the
first convolution to the last linear layer. Neither answer is wrong, which is
the uncomfortable part.

**The mechanism explains it.** `find_failing_ops` walks *backward* from the
outputs and reports an op that fails while its own inputs pass. So the tolerance
decides how far back the walk gets. `conv:input_3` is op 1 of 182 and
`linear:var_930` is op 182 of 182: a loose tolerance stops at the output layer,
a strict one reaches the origin. Wall clock is the giveaway — 62 seconds at
`atol` 1e-3 against **0.7 seconds** at 0.1, because the search terminates
immediately.

Two things generalise:

- **Report the sweep, not the point.** A localization result without the
  tolerance that produced it is not a finding; it is one sample of a curve.
- **"Stabilizes" needs an operational definition.** Here it is *the answer stops
  changing*: the widest run of consecutive tolerances giving an identical op
  set. Not "the count is small" — a confidently wrong single answer also has a
  small count.

### Step 9 — Ask whether your detector can fire at all

The next task was Apple's NaN/Inf scanner, behind the rule *"NaN or Inf in the
output is an unconditional failure, no tolerance debate."* Run it on a normal
input: **zero failing ops, 0.4 seconds.** That looks like a pass.

It is not a pass. A detector that never fires and a model that never breaks
produce identical reports. So before believing the clean result, feed it
something that *should* trip it.

This is a **positive control**, and it is the mirror image of the manipulation
check in §4.5. A manipulation check asks *did my experiment change anything?* A
positive control asks *can my instrument detect anything?* Both defend against
the same failure — a null result that means nothing.

The control was an all-NaN input, and PyTorch was the reference (F-013):

| Input | PyTorch fp32 | Core ML |
|---|---|---|
| all NaN | **1000 NaNs** | finite, **top-1 = 904** |
| all +Inf | **1000 NaNs** | finite, top-1 = 904, *identical to above* |
| 100 NaNs in 150k values | **1000 NaNs** | finite, **top-1 = 111 — same as nominal** |

PyTorch poisons the output, as IEEE arithmetic requires. Core ML returns a
confident, plausible answer. Across 56 configurations the NaN scan **never fired
once**; the Inf scan fired twice, in one corner of the matrix.

So the gate is a sound *sufficient* condition and a nearly useless *necessary*
one, and the real failure mode is worse than a crash: **silent saturation to a
finite wrong answer**. A pipeline bug that feeds garbage to this model in
production gets a class prediction back, not an error.

**Then chase it into the other language.** The finding had an obvious hole —
maybe the Python wrapper was cleaning the input, in which case a shipped app
would behave differently. Driving the same tensors through Swift, which touches
the framework directly, returned *identical* values: top-1 904, |max| 6.592
(F-015). The wrapper was exonerated and the behaviour is the runtime's, which is
the answer that actually matters. **A finding is not finished while a plausible
alternative explanation is still standing** — §4.5 again, applied to a defect
rather than a null result.

### Step 10 — Read the mechanism instead of inferring it

Earlier work noticed that `.all` and `.cpuAndNE` produced bit-identical output
across 200 inputs and concluded Core ML must be dispatching to the Neural
Engine. Reasonable — and still an inference from behaviour.

`MLComputePlan` reports the assignment directly, before anything runs (F-014):

| Model | `.cpuOnly` | `.cpuAndGPU` | `.cpuAndNE` | `.all` |
|---|---|---|---|---|
| **fp16** | 100% CPU | 100% GPU | **100% ANE** | **100% ANE** |
| **fp32** | 100% CPU | 100% GPU | **100% CPU** | **100% GPU** |

The fp16 plans for `.all` and `.cpuAndNE` are *the same plan*. So those were
never two paths that happened to agree — which is a stronger statement than the
inference, and it means the differential between them is really asserting *that
dispatch has not changed*.

It also answered a question that had been left dangling: an fp32 model under
`.cpuAndNE` compiles to a pure-CPU plan, because **no fp32 op can reach the
Neural Engine at all**. A parity suite running fp32 under `.cpuAndNE` is
silently testing the CPU twice.

**Where a direct measurement exists, prefer it to a behavioural inference** —
not because the inference was wrong, but because the direct reading answers
questions the inference cannot.

*And a warning about direct readings.* `supported_compute_devices` reports what
is available **under the configuration you asked for**, not a property of the
op. Under `.cpuOnly` no op lists the Neural Engine — trivially true, and
meaningless. The first version of that script duly reported "9 op types are not
ANE-capable" for every CPU and GPU row, which reads exactly like a finding and
is an artifact of asking a question that configuration cannot answer. **An API
that always returns something will happily answer a question you should not have
asked.**

### Step 11 — Put the findings where they cannot be ignored

A finding in a document gets read once. The last step is moving each one into a
place that enforces it.

- Thresholds in the Swift suite are **quoted from the measurement**: the ANE
  gate is 0.030 because F-008 observed a maximum of 0.0242, and the policy's
  `rationale` field says so in the source.
- `rationale` is **non-optional**. A tolerance policy cannot be constructed
  without stating why those numbers — documentation as a property of the type
  system rather than a review convention.
- Re-blessing a golden **requires a reason**, and the previous reason is carried
  forward rather than overwritten. A golden re-blessed three times says so.
- The most useful one is a test that was deliberately *not* written. fp16 parity
  against PyTorch does **not** assert top-1, because F-009 measured 2–4% top-1
  flips — a top-1 gate there would fail on 2–4% of inputs by construction. That
  is a flake manufactured on purpose. Top-1 is asserted where it actually holds,
  across compute units.

**Knowing which assertion not to write is the same skill as knowing which one to
write**, and it is only available to someone who measured first.

---

## 4. Eight lessons that generalise

### 4.1 Choosing the metric *is* the test design

Look at that table again. Cosine similarity never drops below 0.9995 — by that
measure everything is fine. **Cosine alone would have missed every flip.**

Different statistics are blind to different failures: cosine sees direction,
top-1 sees the user-visible answer, L1 sees magnitude. This is why
`TolerancePolicy` offers several criteria, and why each test must state which it
chose and why. In the Swift design that rationale is a **required, non-optional
field** — a policy cannot be constructed without it, which makes the
documentation a property of the type system rather than a review convention.

### 4.2 Thresholds are measured, not chosen

The ANE gate here is 0.030 because the observed maximum was 0.0242. *"I set the
gate from measured variance rather than picking a number"* is the difference
between a threshold that holds and one that gets widened the first time it fires.

And the corollary: **never widen a threshold to make a test green.** A failing
gate gets triaged to one of three outcomes — explained, fixed, or accepted into
the baseline *with a recorded reason*. Silently relaxing it is the specific
failure the process exists to prevent.

### 4.3 A passing test can be answering the wrong question

F-003 validated the `xcodebuild` workflow in a clean scratch directory. It
passed. It then broke in the real repository, because the real repository
contained a leftover `.xcodeproj` that shadows `Package.swift` (F-004). The
experiment was internally valid; it simply was not testing the same environment.

This is **test validity** — the gap between what a test appears to check and what
it actually checks. It is the deepest failure mode in the discipline precisely
because it is indistinguishable from success.

### 4.4 Flakiness is worse than absence

A test that fails randomly teaches people to re-run it, and a suite people re-run
has stopped gating anything. This is why quarantine (keeping a known-bad test
*running and reporting* without failing the build) beats disabling it, and why
the retry rate is tracked as a metric rather than hidden.

The zero noise floor from F-008 matters here too: these differential tests
**cannot flake for numerical reasons**, so tolerances can be tight rather than
padded defensively.

### 4.5 A negative result is a result — if the manipulation is checked

F-009 reproduced nothing. It is still one of the more useful measurements in the
repository, because it permanently removed one of three explanations, and did so
in the direction that argues *against* the hypothesis rather than merely failing
to support it.

But a null result is only interpretable if you can show your manipulation did
something. Otherwise *"input distribution doesn't matter"* is indistinguishable
from *"my generator didn't actually change the inputs"* — and those two have
opposite implications.

That is what the `logit |max|` row is for: **5.38 → 6.52**. The two input sets
demonstrably put the network at different operating points, so the flat ratio is
evidence about the world rather than evidence about the experiment.

**Report the manipulation check next to the null result, always.** Without one, a
negative finding is unfalsifiable, and unfalsifiable findings persuade nobody who
is paying attention.

### 4.6 "Expected" is not "verified"

The torch-version elimination (F-010) is obvious in hindsight — torch's job ends
at tracing the graph, and the arithmetic is Core ML's. It was worth twenty
minutes anyway, for two reasons.

coremltools itself warns that the installed pairing is untested, which makes it a
live objection anyone could raise against every number in this repository. And
the environment built to settle it is what then produced the diagnostic that
root-caused a crash open since Step 4.

Cheap eliminations are worth running even when you are confident of the outcome.
Confidence is not a measurement, and the byproducts are frequently the point.

### 4.7 "Reproducible", "deterministic" and "portable" are three different claims

They get used interchangeably and they fail independently.

An accidental re-export two days later reproduced every weight and every
measurement **byte for byte** — but the `.mlpackage` around them did not, because
it carries fresh UUIDs and an embedded date stamp on every export (F-011). The
compute plan shows the same split: device assignment is bit-stable across loads
while the estimated cost weights wobble in the last ulp (F-014).

The pattern is worth naming: **the load-bearing part is reproducible, an
incidental float or identifier beside it is not.** It has a direct consequence
for snapshot tests — a golden keyed on a digest of the whole artifact would
invalidate on every export while the numerics were untouched, and the failure
would look exactly like a real model change. So golden identity here is a digest
of the *weights*, never of the package.

And none of that is **portability**. Everything above holds on one machine with
one seed. Whether another chip produces the same weights is a different question
— the one the #2625 investigation could not close.

### 4.8 Regenerating is not reproducing

The Python experiments build their inputs from a seed, which is reproducible and
costs nothing to store. The obvious move is to do the same in Swift.

It does not work. `torch.rand` with seed 0 is a specific generator; Swift's is a
different one. Two "identical seeded random inputs" are entirely different
tensors, and any elementwise comparison against a PyTorch reference computed on
the other one is meaningless.

The tell was in F-015. The malformed inputs — all-NaN, all-Inf — matched across
languages to the digit, because *those are the same tensor however you generate
them*. The nominal inputs did not, and could not.

So the cross-language fixtures are **committed bytes**, not a seed: four inputs
and their references, 2.3 MB, read by both sides and verified against a digest
on load. Volume stays where it is cheap — the Python side runs n=100 to n=200 —
because these fixtures exist for *exactness*, not for coverage.

**Whenever a contract crosses a boundary — two languages, two machines, two
teams — the artifact is the contract, and a recipe for regenerating it is not.**

---

## 5. Vocabulary

| Term | Meaning |
|---|---|
| **Oracle** | The source of truth an assertion compares against |
| **Differential testing** | Using a second implementation or execution path as the oracle |
| **Metamorphic relation** | A property that must hold between related inputs |
| **Golden / snapshot test** | Comparison against a recorded reference output |
| **Tolerance policy** | The documented rule for how much drift is acceptable, and measured how |
| **Noise floor** | The system's disagreement with itself; the lower bound on what any threshold can detect |
| **Control** | A variant included to isolate one variable — here, the fp32 export |
| **Confound** | An alternative explanation for a result that the experiment cannot rule out |
| **Manipulation check** | Evidence that the variable you meant to change actually changed |
| **Positive control** | Evidence that your *detector* can fire — the mirror of a manipulation check |
| **Elimination** | Localising a cause by closing every alternative you can name |
| **Localization** | Attributing a failure to a specific op or layer; here, a function of the tolerance passed |
| **Reproducible** | Same inputs, same code, same machine → same bytes |
| **Portable** | Same inputs, same code, *different* machine → same bytes. A strictly stronger claim |
| **Silent saturation** | Failing to a finite, confident, wrong answer instead of to NaN or a crash |
| **Flake** | A test that both passes and fails on identical input |
| **Quarantine** | Keeping a known-bad test running and reporting without failing the build |
| **Test validity** | Whether a test measures what it claims to measure |
| **Bisection** | Narrowing a failure by varying one factor at a time |
