# Contributing

Use the development dependencies in `uv.lock` and run checks from the repository root:

```bash
uv run ruff check .
uv run ruff format --check .
JAX_PLATFORMS=cpu uv run pytest
```

Describe the behavior changed and the validation performed. Keep model parameters, dataset transforms, and the real-robot protocol explicit. Add tests for changes to action-field supervision, ODE integration, LoRA parameter handling, or robot state and action ordering.

Test-only model configurations belong inside tests; the public registry contains the four multi-task training recipes. Full-model and external-asset checks use the `manual` marker.

Standalone checks for model shapes, LoRA parameters, and the service protocol are in `tests/manual/`. Run them explicitly with `uv run tests/manual/<script>.py`; full-model and checkpoint checks use the arguments documented by their CLI help.
