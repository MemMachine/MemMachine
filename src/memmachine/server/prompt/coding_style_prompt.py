"""Prompt templates for coding-style-focused semantic categories."""

from memmachine.semantic_memory.semantic_model import (
    SemanticCategory,
    StructuredSemanticPrompt,
)

coding_style_tags: dict[str, str] = {
    "Preferred Languages & Versions": "Programming languages, runtimes, and versions the user prefers (e.g., Python 3.12, modern C++, Rust).",
    "Formatting & Style Guides": "Indentation, line length, brace style, naming conventions, and any explicit preferences about PEP8 or other style guides.",
    "Typing & Strictness Preferences": "Attitude toward type hints, static typing, mypy/pyright/TS strictness, and whether they favor strict or loose typing.",
    "Error Handling & Logging Style": "How the user likes to handle errors (exceptions vs return codes), logging level/structure, and failure-handling philosophy.",
    "Testing & TDD Habits": "Preferences for unit/integration tests, frameworks (pytest, JUnit, etc.), coverage expectations, and whether they favor TDD.",
    "Abstraction & Architecture Preferences": "Preference for functional vs OO vs procedural style, use of patterns, modularization, layering, and separation of concerns.",
    "Readability vs Performance Tradeoffs": "Whether the user tends to favor clarity/readability or micro-optimizations/performance in their code.",
    "Documentation & Comments Style": "Docstring formats (Google, NumPy, Sphinx), comment density, and when they think comments are appropriate.",
    "Tooling & Ecosystem Preferences": "Linters, formatters, IDEs, build tools, and other dev tooling the user prefers (e.g., ruff, black, clang-format, uv, poetry).",
    "Concurrency & Parallelism Preferences": "Preferred concurrency models (async/await, threads, processes, coroutines, actors) and any patterns or libraries they like/hate.",
    "Dependency Management & Packaging": "How the user prefers to manage dependencies, virtual envs, packaging, and version pinning.",
    "API & Interface Design Style": "Preferences for function signatures, class APIs, immutability, fluent interfaces, builder patterns, and configuration style.",
    "Data Structures & Library Choices": "Go-to data structures and libraries (e.g., Polars vs pandas, STL vs custom containers) and when they choose each.",
    "Refactoring & Technical Debt Attitude": "How aggressively they like to refactor, their tolerance for technical debt, and attitudes toward quick hacks vs long-term design.",
    "Safety, Robustness & Security Expectations": "Expectations around input validation, defensive programming, security practices, and failure modes.",
}

coding_style_description = """
    Extract all information related to the user's coding style and engineering preferences.

    Focus on:
    - How they like code to look (formatting, naming, structure).
    - How they like code to behave (error handling, robustness, performance vs readability).
    - How they like to work (testing habits, tooling, refactoring, dependency management).
    - Any explicit likes/dislikes about languages, frameworks, or patterns.

    Include implicit preferences inferred from praise, complaints, or repeated patterns in their code
    or feedback about code (e.g., “too much magic”, “too many globals”, “hate single-letter variables”).

    Capture even small details that can help future code be written in their preferred style.
"""

CodingStyleSemanticCategory = SemanticCategory(
    name="coding_style",
    prompt=StructuredSemanticPrompt(
        tags=coding_style_tags,
        description=coding_style_description,
    ),
)

SEMANTIC_TYPE = CodingStyleSemanticCategory
