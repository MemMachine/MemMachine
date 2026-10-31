# Contributing to MemMachine

Thank you for your interest in contributing to MemMachine! We are a
community-driven open-source project, and we welcome contributions from
everyone.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and
[Governance](GOVERNANCE.md) before you start.

## Issues

Check the [issues page](https://github.com/MemMachine/MemMachine/issues) to see
whether your bug or idea has already been reported. For a new bug, a feature,
or a major documentation change, open an issue with the matching template
first, so the change can be discussed before work starts. To take an existing
issue, leave a comment on it. Small fixes, such as typos and broken links, can
go straight to a pull request.

## Setup

1. Fork the repository on GitHub, clone your fork, and add the original
   repository as the `upstream` remote:

   ```bash
   git clone https://github.com/YOUR_USERNAME/MemMachine.git
   cd MemMachine
   git remote add upstream https://github.com/MemMachine/MemMachine.git
   ```

2. Install [uv](https://docs.astral.sh/uv/). From the repository root, install
   every package in editable mode, with all optional dependencies and the
   development tools:

   ```bash
   uv sync --all-extras
   ```

## Checks

CI runs these checks on every pull request that is not a draft. Run them from
the repository root before you push.

- Format and lint with [Ruff](https://docs.astral.sh/ruff/):

  ```bash
  uv run ruff format
  uv run ruff check
  ```

- Type check each package with [ty](https://docs.astral.sh/ty/):

  ```bash
  uv run ty check --project packages/common
  uv run ty check --project packages/client
  uv run ty check --project packages/server
  ```

- Run the unit tests with pytest, and add tests for your change:

  ```bash
  uv run pytest packages/server/server_tests packages/client/client_tests
  ```

  Integration tests are marked `integration`; run them by adding
  `-m integration`. Most start their databases in Docker. The rest connect to
  a server you run, and skip when they can't reach it:

  - NebulaGraph Enterprise 5.0 or later, at `NEBULA_HOST` (default
    `127.0.0.1:9669`), with `NEBULA_USER` and `NEBULA_PASSWORD`
  - An OpenAI-compatible server serving `qwen3:8b`, such as Ollama, at the
    base URL in `OLLAMA_HOST`, which has no default
  - A MemMachine server, for the Python client's tests, at
    `MEMORY_BACKEND_URL` (default `http://localhost:8080`)

  Tests that call OpenAI, Amazon Bedrock, or Cohere use the real APIs, billed
  to the keys you give them. They read test-only variables and run only when
  those are set:

  - OpenAI: `MEMMACHINE_TEST_OPENAI_API_KEY`
  - Amazon Bedrock: `MEMMACHINE_TEST_AWS_ACCESS_KEY_ID`,
    `MEMMACHINE_TEST_AWS_SECRET_ACCESS_KEY`, and `MEMMACHINE_TEST_AWS_REGION`,
    plus `MEMMACHINE_TEST_AWS_SESSION_TOKEN` for temporary credentials
  - Cohere: `MEMMACHINE_TEST_COHERE_API_KEY`

  [Running PyTest Unit Tests](maintainers/running-unit-tests.md) shows more ways
  to select tests.

The TypeScript client in `packages/ts-client/` has its own commands, listed in
its [README](packages/ts-client/README.md#development).

## Style

Ruff and ty check formatting, lint rules, and types. Beyond what they check,
follow the style of the surrounding code and these rules:

- **Names:** spell names out (`derivative_uuid`, not `duuid`), private names
  included. Functions that share a prefix behave alike: `is_*` returns a
  bool, and `validate_*` raises. Use American spelling in names and text.
- **Types:** don't use `Any`, `cast`, or `object` as a type unless a
  third-party library's types leave no other way. Narrow the value or change
  the code instead.
- **Suppressions:** don't add `# noqa`, `# type: ignore`, or `# ty: ignore`,
  except where a third-party library's API forces one. Fix the code, or
  change the rule in `pyproject.toml` if it is wrong for the whole project.
- **Libraries:** use only their public, documented API. In async code, use a
  library's async client when it has one. Declare every package your code
  imports as a dependency, and import an optional extra's modules only in
  the code that needs them.
- **Errors:** catch broad exceptions only at API boundaries, and log them
  with context.
- **Comments:** explain what the code can't say itself: why a choice is
  safe, an ordering or concurrency constraint, a pitfall. State what holds
  now. Measurements, history, and references to issues or pull requests go
  in the commit message and the pull request description. A comment
  describes its own code and what that code uses, not the code that uses it.
- **Docstrings:** a docstring gives the contract from the caller's side. An
  abstract method's docstring says when it is called, what implementations
  must guarantee, and what they may raise, without naming an
  implementation. A method marked `@override` has no docstring; notes on how
  it works go in a comment in its body.
- **Structure:** factor out duplicated code only when the shared piece is
  one unit whose name says everything it does, and the callers read better
  for it. Don't wrap a one-line call in a helper. Before caching or
  memoizing, weigh the work saved against the state it adds, which has to be
  maintained and kept correct.
- **Tests:** mark tests that need an external service `integration` and
  slow tests `slow`.

Existing code may not match these rules; that is no license for new code to
break them.

## Documentation

The documentation site's source is in `docs/`. [docs/README.md](docs/README.md)
explains how to preview it locally.

## Pull Requests

1. Create a branch:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Sign every commit with `-sS`, which adds a `Signed-off-by` line and a
   cryptographic signature. GitHub's guide to
   [signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)
   explains how to set up a signing key.

   ```bash
   git commit -sS -m "Add a new feature"
   ```

3. Push the branch to your fork and open a pull request against `main`, filling
   in the template. A draft pull request does not run CI; the checks run once
   you mark it ready for review.

We look forward to your contributions!
