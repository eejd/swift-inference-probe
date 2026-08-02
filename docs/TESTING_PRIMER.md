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

---

## 4. Four lessons that generalise

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
| **Flake** | A test that both passes and fails on identical input |
| **Quarantine** | Keeping a known-bad test running and reporting without failing the build |
| **Test validity** | Whether a test measures what it claims to measure |
| **Bisection** | Narrowing a failure by varying one factor at a time |
