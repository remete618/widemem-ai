# Contributing to widemem

Thanks for considering a contribution. Here's how to get started.

## Setup

```bash
git clone https://github.com/remete618/widemem-ai.git
cd widemem
python3 -m pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

All tests use mocks — no API keys or external services needed.

## Code style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Run `ruff check .` and `ruff format .` before submitting
- No docstrings or comments unless the logic isn't self-evident
- No type stubs or backwards-compatibility shims — just change the code
- Keep it simple. If three lines work, don't write an abstraction

## Pull requests

1. Fork the repo and create a branch from `main`
2. Write tests for any new functionality
3. Make sure all tests pass (`pytest`)
4. Make sure linting passes (`ruff check .`)
5. Keep PRs focused — one feature or fix per PR
6. Write a clear PR description explaining what and why

## What we're looking for

- Bug fixes with a test that reproduces the bug
- New provider implementations (LLM, embedding, vector store)
- Performance improvements with benchmarks
- Documentation improvements

## What we're not looking for (yet)

- Major architectural changes without prior discussion — open an issue first
- Features that add complexity without clear user value
- Dependencies on large frameworks or libraries

## Reporting bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Be decent.
