from __future__ import annotations

from pathlib import Path

from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec

SPEC_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "memmachine_server"
    / "retrieval_skill"
    / "skills"
    / "specs"
    / "sub_skills"
)


def _raw_text(file_name: str) -> str:
    return (SPEC_ROOT / file_name).read_text(encoding="utf-8")


def _policy_markdown(file_name: str) -> str:
    spec = load_skill_spec(SPEC_ROOT / file_name)
    assert spec.policy_markdown is not None
    return spec.policy_markdown


def _assert_section_order(policy: str) -> None:
    section_order = [
        "## Intent",
        "## Rules",
        "## Tools",
        "## Output Contract",
        "## Examples",
        "## Failure Modes",
    ]
    previous_idx = -1
    for heading in section_order:
        current_idx = policy.find(heading)
        assert current_idx >= 0, f"missing required section heading: {heading}"
        assert current_idx > previous_idx, (
            f"section order violation: {heading} appeared out of order"
        )
        previous_idx = current_idx


def _assert_contains_all(policy: str, snippets: list[str]) -> None:
    for snippet in snippets:
        assert snippet in policy, f"missing policy anchor: {snippet}"


def test_fixed_section_order_is_consistent_across_sub_skills() -> None:
    for file_name in ("coq.md",):
        policy = _policy_markdown(file_name)
        _assert_section_order(policy)


def test_sub_skills_expose_examples_and_failure_modes() -> None:
    for file_name in ("coq.md",):
        policy = _policy_markdown(file_name)
        assert policy.count("### Example") >= 3
        assert policy.count("### Example") <= 6
        assert "## Failure Modes" in policy


def test_coq_prompt_parity_anchors_are_preserved() -> None:
    policy = _policy_markdown("coq.md")
    _assert_contains_all(
        policy,
        [
            "Use retrieved memory as the primary source of knowledge.",
            "Run at least one `memmachine_search` before any final sufficiency decision.",
            "Do not invent new entities.",
            "Strict sufficiency standard",
            "If uncertain, choose `is_sufficient=false`.",
            "Next-best rewritten query objective",
            "earliest blocking hop",
            "Avoid duplicates of tried rewritten queries after normalization.",
            "Confidence calibration",
            "evidence_indices",
        ],
    )


def test_coq_output_contract_is_strict_v1_and_fail_closed() -> None:
    policy = _policy_markdown("coq.md")
    _assert_contains_all(
        policy,
        [
            "`v1` required fields",
            "`is_sufficient`: boolean",
            "`evidence_indices`: array of integer indices (0-based, no negatives)",
            "`new_query`: single-line string",
            "`confidence_score`: number in `[0.0, 1.0]`",
            "set `new_query` to `original_query` exactly",
        ],
    )


def test_sub_skills_do_not_reintroduce_placeholder_language() -> None:
    disallowed_snippets = [
        "translated from legacy",
        "placeholder summary",
        "summary placeholder",
        "todo: translate",
    ]
    for file_name in ("coq.md",):
        lower_text = _raw_text(file_name).lower()
        for snippet in disallowed_snippets:
            assert snippet not in lower_text


def test_sub_skills_keep_explicit_v1_contract_markers() -> None:
    for file_name in ("coq.md",):
        policy = _policy_markdown(file_name)
        assert "`v1` required fields" in policy
        assert "Fail-closed requirements" in policy
