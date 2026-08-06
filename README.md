# InferenceProbe

InferenceProbe is a Core ML inference-validation harness for Apple silicon. It
tests numerical agreement across CPU, GPU, and Neural Engine execution,
precision drift, robustness, and eventually performance regressions. It tests
the inference runtime rather than the model's research quality.

## Current state

M0–M3 are complete and M4 is substantially so: seven Python experiments,
**fifteen recorded findings**, and **31 Swift tests across 7 suites** green under
both `swift test` and `xcodebuild`. Remaining: metamorphic relations, the
latency percentile harness, and the XCTest performance suite. See:

- [`docs/ROADMAP.md`](docs/ROADMAP.md) for milestone status and build order.
- [`docs/FINDINGS.md`](docs/FINDINGS.md) for measured results and judgements.
- [`docs/SPEC.md`](docs/SPEC.md) for architecture and component contracts.
- [`docs/TEST_STRATEGY.md`](docs/TEST_STRATEGY.md) for the oracle and tolerance
  policy.

## Python quick start

The primary conda environment is already installed on the development machine.
Activate it and run the scripts from `python/` so the bare script names resolve.
Their default input and output paths are anchored to the repository itself, so
they read and write `Fixtures/models/` no matter where they are invoked from:

```bash
cd path/to/InferenceProbe/python
conda activate inferenceprobe

python --version
python -c 'import torch, torchvision, coremltools; print(torch.__version__, torchvision.__version__, coremltools.__version__)'
```

Expected versions are Python 3.12.13, torch 2.13.0, torchvision 0.28.0, and
coremltools 9.0. If `conda activate` is unavailable in a fresh zsh terminal:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate inferenceprobe
```

coremltools prints a warning that torch 2.13.0 is newer than its tested pairing.
That warning is expected; finding F-010 separately verifies the experiment with
torch 2.7.0.

`conda init zsh` makes initialization persistent for future terminals. Once the
environment is active, the implemented experiments are:

```bash
python export_model.py               # export fp32 + fp16, drift vs PyTorch   (F-005, F-006)
python compute_unit_differential.py  # noise floor + 4 compute units, n=200   (F-007, F-008)
python repro_2625.py                 # fp16 drift across input distributions  (F-009, F-010)
python localize.py                   # which op diverges, swept across atol   (F-012)
python validate.py                   # NaN/Inf scan + malformed-input battery (F-013)
python compute_plan.py               # per-op device assignment, ANE residency (F-014)
python generate_goldens.py           # commit input tensors + PyTorch references
```

`generate_goldens.py` writes to `Fixtures/inputs/` and `Fixtures/goldens/`,
which **are** tracked — unlike the models, they cannot be regenerated
identically in another language (F-015), so the bytes are the contract the
Swift `ParityTests` read. Re-running it re-blesses; pass `--reason` to say why.

Each writes its result JSON into `Fixtures/models/`, and those files are
committed — so the useful check is not just that a script runs but that it
reproduces what is already there.

`localize.py` is the slow one at roughly 15 minutes, because a strict `atol`
makes it traverse the whole op graph. The rest are a few minutes or less. For a
quick smoke run rather than a full measurement:

```bash
python localize.py --arm compute --atols 1e-2      # one arm, one tolerance
python compute_unit_differential.py --n 20 --repeats 5
python repro_2625.py --n 10
python validate.py --units cpuOnly --no-torch-reference
```

Pass `--out` to any of them to write somewhere else and leave the committed
fixtures alone.

### Reproducing, rather than just running

`compute_unit_differential.py`, `repro_2625.py`, and `localize.py` write no
timestamps, so re-running them should reproduce the committed JSON byte for
byte (F-011). That makes this a real end-to-end check:

```bash
python compute_unit_differential.py && python repro_2625.py
git -C .. diff --stat Fixtures/                           # expect: no output
```

**Two scripts are exceptions, both because they stamp a timestamp:**

- `export_model.py` writes `exported_at`, so re-running always dirties the two
  manifest files even when every weight is identical. Run it only to regenerate
  the `.mlpackage` files — gitignored, and therefore mandatory on a fresh clone
  before anything else works.
- `generate_goldens.py` writes `blessedAt`, so re-running dirties the golden
  manifests. The tensors themselves are unchanged. Re-run it only when you mean
  to re-bless, and pass `--reason` saying why — that is the point of the field.

Use `conda deactivate` when finished. `inferenceprobe-torch27` is an isolated
comparison environment for finding F-010 and is not the normal development
environment.

## Swift quick check

```bash
cd path/to/InferenceProbe
swift test
```

Generated `.mlpackage` files are intentionally ignored because they are large
and reproducible. Their version manifests and measurement JSON are tracked in
`Fixtures/models*`.

**Suites that need a model skip rather than fail** when it is absent, naming the
regeneration command in the skip message. So a fresh clone runs the harness,
comparison-arithmetic, and fixture-integrity suites green and skips the rest —
which is exactly what CI does.

## CI

Two workflows, and only one of them is a gate.

**`pr.yml`** runs on every push and pull request. It uses `xcodebuild`, not
`swift test`, because only the former emits the `.xcresult` the reporting layer
consumes (F-003). It runs the chip-independent suites and reports skips as a
first-class number rather than folding them into "passed".

**`chip-differential.yml`** is an experiment, not a gate — nothing merges on its
result. Every threshold in this repository is **M4 Pro-derived**, and F-010
concluded by elimination that the unexplained gap to coremltools #2625 is
attributable to hardware, while recording that confirming it "would need the
same script run on different silicon." A GitHub macOS runner *is* different
silicon, so this workflow runs the pinned experiments there and prints the
numbers beside the committed ones. `repro_2625.py` uses `.cpuOnly` only, so it
does not depend on a Neural Engine being exposed inside the runner VM — which is
undocumented and may well be absent, itself a useful thing to learn.
