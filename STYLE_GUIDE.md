
# MemMachine Code Style Guide

Consistency in code style is crucial for readability and collaboration. This guide outlines the core principles for all contributions to the MemMachine project and points to specific, automated style enforcement tools for each language.

## Core Principles

- **Readability:** Code should be easy to read and understand. Favor clarity over cleverness.
- **Consistency:** Follow the established style and formatting of the surrounding code. When in doubt, follow the recommendations in this guide.
- **Clarity:** Use meaningful names for variables, functions, and classes. Avoid abbreviations unless they are standard and widely understood.
- **Documentation:** All public APIs, complex functions, and non-obvious code sections should be well-commented.

## Language-Specific Style Guides & Tools

We use automated tools to enforce our style guides. Please install and run these tools before submitting a pull request.
They can be installed easily as part of the `dev` group using `uv sync`.

### Python
- **Formatter:** [**Ruff**](https://github.com/astral-sh/ruff)
  - Ruff is an extremely fast code formatter that automatically formats Python code.
- **Linter:** [**Ruff**](https://github.com/astral-sh/ruff)
  - Ruff is an extremely fast linter that is designed to be a drop-in replacement for tools like flake8 and isort. It catches common bugs and stylistic issues.
- **Type Checker:** [**Mypy**](https://github.com/python/mypy)
  - Mypy is a static type checker for Python.

### Markdown

- **Style Guide:** We follow a consistent style for our Markdown files to ensure the documentation is easy to read and maintain.
  - **Line Length:** Limit lines to 80 characters for readability, especially when viewing diffs.
    - **Headings:** Use `#` for top-level headings and subsequent headings `##`, `###`, etc., for subheadings.
    - **Code Blocks:** Use triple backticks (```) with a language identifier (e.g., `python`, `javascript`) for all code blocks.
    - **Emphasis:** Use a single asterisk (`*`) for italics and a double asterisk (`**`) for bold.
    - **Links:** Use meaningful, descriptive link text instead of "click here."
    - **Lists:** Use hyphens (`-`) for unordered lists and consistent numbering for ordered lists.
- **Linter:** [**markdownlint**](https://github.com/DavidAnson/markdownlint)
  - Use markdownlint to check for style and format issues in Markdown files.

## Formatting

- **Automated Formatting:** We use automated tools to format our code. Please run the formatter (e.g., Ruff for Python) before committing to ensure all code is correctly styled.
- **Linting:** Our CI pipeline runs a linter (e.g., Ruff or ESLint) on every pull request. Your changes must pass these checks to be merged.
- **Editor Configuration:** We highly recommend configuring your editor (e.g., VS Code) with the appropriate extensions to automatically format and lint code on save. This makes following the style guide effortless.
