# MemMachine Agent Guide

This file helps autonomous coding agents work effectively in this repo. It
summarizes build/lint/test commands and the code style expectations for both
Python and the TypeScript REST client packages.

## Repository Orientation

- `packages/server/`: server Python package (server, SDK, memory layers).
- `packages/common/`: common Python package (shared REST API).
- `packages/client/`: client Python package.
- `packages/ts-client/`: client TypeScript package.
- `packages/meta/`: meta Python package (both server and client).
- `integrations/`, `examples/`, `evaluation/`, `tools/`: demos and helpers.
- `STYLE_GUIDE.md`: higher-level style notes and tooling references.

## Environment Setup

- Recommended: `uv sync` from repo root.
- With optional extras: `uv sync --all-extras`.
- Python version: 3.12+ (see `pyproject.toml`).

## Build, Lint, and Test Commands

### Python (repo root)

- Install deps: `uv sync`.
- Lint: `uv run ruff check` (or `ruff check`).
- Format: `uv run ruff format` (or `ruff format`).
- Type check: `uv run ty check packages` (or `ty check packages`).
- Run all tests: `uv run pytest` (or `pytest`).
- Run a single test file:
  `uv run pytest packages/server/server_tests/memmachine_server/common/test_utils.py`.
- Run a single test function:
  `uv run pytest packages/server/server_tests/memmachine_server/common/test_utils.py::test_chunk_text`.
- Run tests by keyword:
  `uv run pytest -k "create_memory"`.
- Run marked tests only:
  `uv run pytest -m integration` or `uv run pytest -m slow`.

### Docker Image Build

- Build/push container images: `./build-docker.sh` (interactive) or use the
  flags documented in `build-docker.sh`.

### TypeScript REST Client (per package)

Paths with package.json:
- `packages/ts-client/`

Common commands (run from one of the above directories):

- Install deps: `npm install`.
- Build: `npm run build`.
- Lint: `npm run lint` (or fix with `npm run lint:fix`).
- Format: `npm run format` (or `npm run format:check`).
- Tests: `npm run test`.
- Single Jest test by name: `npm run test -- -t "test name"`.
- Single Jest test file: `npm run test -- path/to/file.test.ts`.

## Code Style Guidelines

### General Principles

- Readability over cleverness; mirror nearby patterns.
- Favor small, testable functions with clear responsibilities.
- Keep public APIs and non-obvious logic documented.

### Python Formatting and Imports

- Formatting is enforced by Ruff. Run `ruff format` before committing.
- Imports are sorted by Ruff (stdlib → third-party → local).
- Avoid wildcard imports; keep import lists minimal and explicit.

### Python Types and APIs

- Use type hints everywhere; prefer `list[str]`, `dict[str, int]`, etc.
- Use `typing.Any` only when unavoidable; document why.
- Prefer `pydantic` models for structured inputs/outputs.
- Keep async boundaries clear; avoid blocking calls in async code.
- Use `async def` + `await` consistently in async modules.

### Python Naming Conventions

- Modules/files: `snake_case.py`.
- Functions/variables: `snake_case`.
- Classes/exceptions: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Tests: `test_*.py` with `test_*` functions.

### Error Handling and Logging

- Raise explicit exceptions with clear messages.
- Preserve context with `raise ... from err`.
- Catch broad exceptions only at API boundaries; log with context.
- Avoid silent failures; return meaningful errors from external I/O layers.

### Testing Expectations

- Use pytest; prefer fixtures for shared setup.
- Keep tests deterministic; isolate external services with mocks/fixtures.
- Mark slow or integration tests using `@pytest.mark.slow` or
  `@pytest.mark.integration`.

### TypeScript REST Client Style

- Linting via ESLint; formatting via Prettier (see package.json scripts).
- Prefer explicit types over `any`; use interfaces/types for DTOs.
- Keep async calls `await`ed; handle errors with typed error paths.
- Use `PascalCase` for types/classes, `camelCase` for functions/vars.

### Documentation (Markdown)

- Keep lines ~80 characters when reasonable.
- Use fenced code blocks with language identifiers.
- Use descriptive link text; avoid bare URLs in prose.
- Use `-` for unordered lists and consistent numbering for ordered lists.

## Notes for Agents

- This repo is a Python-first workspace with optional TypeScript clients.
- If you touch linted files, re-run Ruff to avoid CI failures.
- Avoid reformatting unrelated files; keep diffs scoped to your change.
