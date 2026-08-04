# InferenceProbe

InferenceProbe is a Core ML inference-validation harness for Apple silicon. It
tests numerical agreement across CPU, GPU, and Neural Engine execution,
precision drift, robustness, and eventually performance regressions. It tests
the inference runtime rather than the model's research quality.

## Current state

The Swift package and Python export/measurement pipeline work. M0–M2 are
complete; M3 is partial; the reusable Swift validation suites and CI planned for
M4/M5 are not implemented yet. See:

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
cd /Users/weihantu/InferenceProbe/python
conda activate inferenceprobe

python --version
python -c 'import torch, torchvision, coremltools; print(torch.__version__, torchvision.__version__, coremltools.__version__)'
```

Expected versions are Python 3.12.13, torch 2.13.0, torchvision 0.28.0, and
coremltools 9.0. If `conda activate` is unavailable in a fresh zsh terminal:

```bash
source /Users/weihantu/miniforge3/etc/profile.d/conda.sh
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
```

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
git -C /Users/weihantu/InferenceProbe diff --stat Fixtures/    # expect: no output
```

`export_model.py` is the exception — it stamps `exported_at`, so re-running it
always dirties the two manifest files even when every number is identical. Run
it only when you actually need to regenerate the `.mlpackage` files, which are
gitignored and therefore mandatory on a fresh clone before anything else works.

Use `conda deactivate` when finished. `inferenceprobe-torch27` is an isolated
comparison environment for finding F-010 and is not the normal development
environment.

## Swift quick check

```bash
cd /Users/weihantu/InferenceProbe
swift test
```

Generated `.mlpackage` files are intentionally ignored because they are large
and reproducible. Their version manifests and measurement JSON are tracked in
`Fixtures/models*`.
