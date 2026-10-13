# Research integration tests

The fish tests use Python's standard-library `unittest` runner and the exact
29 scientific plus 5 tooling pins. From the repository root:

```sh
uv venv --python 3.12.8 .venv-research-checks
uv pip install --python .venv-research-checks/bin/python \
  -r experiments/fishy/reference/paper_synth/requirements-paper.txt \
  -r experiments/fishy/requirements-tooling.txt
.venv-research-checks/bin/python -m unittest discover -s tests/fishy -p 'test_*.py' -v
```

Tests create small numerical arrays and disposable files. They need no private
research data, network access, GPU, Blender, VTK or installed InterDeCA core.
They cover strict/frozen boundaries and explicit legacy CSV adapters, preserved
legacy document layouts, finite generator inputs, RNG isolation, comparison
policy and supplemental inference. Stage tests execute the real coordinator
with only the scientific child process mocked; they check failed/interrupted
resume state, dependency invalidation and source/input tamper rejection.

The GitHub Actions research check installs those same pins and runs the authored
tests, the published evidence hash verifier, Ruff 0.16.10 and compilation. Fast tests establish these software contracts;
full paper reproduction and scientific parity require separately recorded runs.
