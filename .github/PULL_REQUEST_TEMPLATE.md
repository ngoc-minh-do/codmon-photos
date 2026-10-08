## What changed

<!-- Short description of the change and why. -->

## Checks

- [ ] `uvx ruff check scripts --select E,F,I,UP,B --ignore E501` passes
- [ ] `uv run python -m py_compile scripts/*.py` passes
- [ ] `README.md` / `CHANGELOG.md` updated if user-visible behavior changed
- [ ] No `.env` or real credentials included

## Test evidence

<!-- Output of a run or test that demonstrates the change (redact credentials). -->