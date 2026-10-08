# Contributing

Thanks for your interest in `codmon-photos`! This is a small personal utility,
so keep changes focused and reviewable.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.14.

```bash
git clone <your-fork-url> codmon-photos
cd codmon-photos
uv sync          # create .venv and install the locked deps
cp .env.example .env   # fill CODMON_EMAIL / CODMON_PASSWORD (and optionally SMB_* / APPRISE_URL)
```

## Local validation

Run the same checks CI runs before pushing:

```bash
uvx ruff check scripts --select E,F,I,UP,B --ignore E501
uv run python -m py_compile scripts/*.py
```

## Opening a PR

1. Create a feature branch from `main` (`git checkout -b feat/my-change`).
2. Make your change, run the validation above, and commit.
3. Open a pull request to `main` and describe what changed and why.

Maintainers squash-merge PRs, so keep the branch history about the change;
individual commit messages should be short imperative sentences
(e.g. `Skip albums whose target folder already exists`).

## What to include

- A short description of the problem and the change.
- For behavior changes: the output of a run or a test that demonstrates it.
- Update `README.md` and/or `CHANGELOG.md` under `[Unreleased]` when the change
  affects user-visible behavior.
- Never commit `.env` files or real credentials — placeholders belong in
  `.env.example`.