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
python export_model.py
python compute_unit_differential.py
python repro_2625.py
```

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
