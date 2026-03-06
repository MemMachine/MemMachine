from __future__ import annotations

from pathlib import Path

import pytest

from memmachine_server.retrieval_skill.skills.spec_loader import load_skill_spec
from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
)

SPEC_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "memmachine_server"
    / "retrieval_skill"
    / "skills"
    / "specs"
)
SUB_SKILL_SPEC_DIR = SPEC_ROOT / "sub_skills"


def _read_sub_skill_spec_text(file_name: str) -> str:
    return (SUB_SKILL_SPEC_DIR / file_name).read_text(encoding="utf-8")


def test_load_markdown_top_level_spec_file() -> None:
    spec_path = SPEC_ROOT / "top_level" / "retrieve_skill.md"
    spec = load_skill_spec(spec_path)

    assert spec.name == "retrieve-skill"
    assert spec.kind == "top-level"
    assert "spawn_sub_skill" in spec.allowed_actions
    assert spec.policy_markdown is not None
    assert "## Actions" in spec.policy_markdown
    assert "direct_memory" in spec.policy_markdown


def test_load_markdown_coq_sub_skill_spec_file() -> None:
    spec_path = SUB_SKILL_SPEC_DIR / "coq.md"
    spec = load_skill_spec(spec_path)

    assert spec.name == "coq"
    assert spec.kind == "sub-skill"
    assert "memmachine_search" in spec.allowed_tools
    assert "return_sub_skill_result" in spec.allowed_tools
    assert spec.policy_markdown is not None
    assert "## Examples" in spec.policy_markdown
    assert "## Failure Modes" in spec.policy_markdown
    assert "evidence_indices" in spec.policy_markdown


def test_sub_skill_markdown_specs_do_not_use_placeholder_language() -> None:
    for file_name in ("coq.md",):
        raw_text = _read_sub_skill_spec_text(file_name)
        lower_text = raw_text.lower()
        assert "translated from legacy" not in lower_text
        assert "placeholder summary" not in lower_text


def test_markdown_spec_missing_required_section_fails(tmp_path: Path) -> None:
    broken_spec = tmp_path / "broken.md"
    broken_spec.write_text(
        """---
name: retrieve-skill
version: v1
kind: top-level
description: broken
route_name: retrieve-skill
required_sections:
  - Intent
  - Rules
  - Actions
  - Completion
---

## Intent
intent text

## Rules
rules text
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillContractError) as exc_info:
        load_skill_spec(broken_spec)

    assert exc_info.value.code == SkillContractErrorCode.INVALID_SPEC.value
    assert "missing required section" in exc_info.value.payload.why.lower()


def test_markdown_spec_unknown_frontmatter_field_fails(tmp_path: Path) -> None:
    bad_frontmatter = tmp_path / "bad-frontmatter.md"
    bad_frontmatter.write_text(
        """---
name: retrieve-skill
version: v1
kind: top-level
description: invalid frontmatter
route_name: retrieve-skill
unknown_field: true
required_sections:
  - Intent
---

## Intent
intent text
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillContractError) as exc_info:
        load_skill_spec(bad_frontmatter)

    assert exc_info.value.code == SkillContractErrorCode.INVALID_SPEC.value
