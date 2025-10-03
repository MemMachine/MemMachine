# Contributing to MemMachine Core

Welcome! We are excited to have you contribute to the core of the MemMachine
project. This guide will help you get started with the codebase and the
contribution process.

## 1. Getting Started: The Issue-First Workflow

Before you begin, please check our
[issues page](https://github.com/MemMachine/MemMachine/issues) to see if the
issue has already been reported.

For a new bug or a feature request, please create a new issue first. This allows
us to discuss the change and ensures that no one else is already working on it.
When creating a new issue, please use the correct template for a bug report or a
feature request.

If you are working on an existing open issue, please leave a comment to let us
know that you are taking ownership of it.

## 2. Local Setup

Follow these steps to set up your local development environment:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/MemMachine/MemMachine.git
    cd MemMachine
    ```

2. **Install the Package:**

    We recommend using [uv](https://docs.astral.sh/uv/) to manage Python environments, versions, and dependencies.

    Alternatively, you may wish to create your own Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

    If using uv, install the package in editable mode
    in an automatically created virtual environment
    from the project root directory, where the `pyproject.toml` file resides:
    ```bash
    uv sync

    # To include all extra dependencies:
    uv sync --all-extras
    ```

    Alternatively, in your own Python environment:
    ```
    pip install -e "."
    # OR
    pip install -e ".[gpu]"
    ```

## 3. Code Style and Quality

The project enforces a strict code style using [Ruff](https://docs.astral.sh/ruff/) for formatting and linting.
The project also uses [Mypy](https://mypy-lang.org/) for type checking.

These tools are installed automatically as part of the `dev` group if using uv.

Alternatively, they may be installed by:
```
pip install ruff
pip install mypy
# OR, if pip is new enough:
pip install --group dev
```

From the project root (where `pyproject.toml` is):

To run the Ruff linter:
```
uv run ruff check
# OR
ruff check
```

To run the Ruff formatter:
```
uv run ruff format
# OR
ruff format
```

To run the Mypy type checker:
```
uv run mypy src
# OR
mypy src
```

## 4. Testing

All contributions should include unit tests to ensure functionality.

- To run the entire test suite, use: `pytest`
- To run tests for a specific file, use: `pytest path/to/your_test_file.py`

## 5. Submitting a Pull Request

Once you have made your changes, follow these steps to submit a pull request:

1. **Create a New Branch:**

    ```bash
    git checkout -b feat/your-feature-name
    ```

2. **Commit Your Changes:**

    ```bash
    git add .
    git commit -sS -m "Feat: Add a new feature for the core"
    ```

    **Remember to use the `-sS` flags to sign your commit.** Unsigned commits
    will fail the GitHub Actions checks.

3. **Push the Branch:**

    ```bash
    git push origin feat/your-feature-name
    ```

4. **Open a Pull Request:** Go to the GitHub repository page and open a pull
request. Complete the pull request template and submit the PR for review.

Thank you for your contribution!
