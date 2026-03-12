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
        "## Completion",
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
        assert "## Failure Modes" in policy


def test_coq_prompt_parity_anchors_are_preserved() -> None:
    policy = _policy_markdown("coq.md")
    _assert_contains_all(
        policy,
        [
            "Always prioritize the earliest blocking hop.",
            "avoid duplicate searches after normalization",
            "Call `memmachine_search` with the new targeted query.",
            "Evaluate using all retrieved evidence, not only the latest search.",
            "answer in plain text directly in the assistant response.",
            "Never emit a return tool or structured summary payload.",
        ],
    )


def test_sub_skills_do_not_reintroduce_legacy_tools() -> None:
    lower_text = _raw_text("coq.md").lower()
    for snippet in ("spawn_sub_skill", "return_sub_skill_result", "return_final"):
        assert snippet not in lower_text
