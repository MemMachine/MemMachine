"""Strict loader for retrieval skill specifications."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from memmachine_server.retrieval_skill.skills.types import (
    SkillContractError,
    SkillContractErrorCode,
    SkillContractErrorPayload,
    SkillSpecV1,
)

DEFAULT_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "top-level": [
        "Intent",
        "Rules",
        "Actions",
        "Completion",
    ],
    "sub-skill": [
        "Intent",
        "Rules",
        "Tools",
        "Output Contract",
    ],
}


def _split_frontmatter(markdown_text: str) -> tuple[dict[str, object], str]:
    if not markdown_text.startswith("---\n"):
        raise ValueError("Markdown spec must begin with YAML frontmatter.")
    end_idx = markdown_text.find("\n---\n", 4)
    if end_idx == -1:
        raise ValueError("Markdown spec is missing frontmatter terminator.")

    frontmatter_text = markdown_text[4:end_idx]
    body = markdown_text[end_idx + 5 :].strip()
    parsed_frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(parsed_frontmatter, dict):
        raise TypeError("Markdown frontmatter must deserialize to an object.")
    return parsed_frontmatter, body


def _validate_required_sections(
    *,
    body: str,
    required_sections: list[str],
) -> None:
    for section in required_sections:
        pattern = rf"(?m)^##\s+{re.escape(section)}\b"
        if not re.search(pattern, body):
            raise ValueError(
                f"Markdown spec missing required section heading: {section}"
            )


def _normalize_required_sections(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("required_sections must be a list of strings.")
    required_sections: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("required_sections entries must be strings.")
        required_sections.append(item)
    return required_sections


def _parse_markdown_spec(text: str) -> dict[str, object]:
    frontmatter, body = _split_frontmatter(text)
    kind_value = frontmatter.get("kind", "inline")
    if not isinstance(kind_value, str):
        raise TypeError("kind must be a string in markdown frontmatter.")

    required_sections = _normalize_required_sections(
        frontmatter.get("required_sections")
    )
    if not required_sections:
        required_sections = DEFAULT_REQUIRED_SECTIONS.get(kind_value, [])
    _validate_required_sections(body=body, required_sections=required_sections)

    spec_data = dict(frontmatter)
    spec_data["required_sections"] = required_sections
    spec_data["policy_markdown"] = body
    return spec_data


def _looks_like_markdown_spec(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("---") and "\n## " in text


def _parse_raw_spec(raw_spec: dict[str, object] | str | Path) -> dict[str, object]:
    if isinstance(raw_spec, dict):
        return raw_spec

    text: str
    if isinstance(raw_spec, Path):
        text = raw_spec.read_text(encoding="utf-8")
    else:
        candidate = Path(raw_spec)
        text = candidate.read_text(encoding="utf-8") if candidate.exists() else raw_spec

    if _looks_like_markdown_spec(text):
        return _parse_markdown_spec(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(text)

    if not isinstance(parsed, dict):
        raise TypeError("Skill spec must deserialize to a JSON/YAML object.")

    return parsed


def load_skill_spec(raw_spec: dict[str, object] | str | Path) -> SkillSpecV1:
    """Load and validate a strict v1 skill spec."""
    try:
        parsed_spec = _parse_raw_spec(raw_spec)
        return SkillSpecV1.model_validate(parsed_spec)
    except ValidationError as err:
        raise SkillContractError(
            code=SkillContractErrorCode.INVALID_SPEC,
            payload=SkillContractErrorPayload(
                what_failed="Skill specification validation failed",
                why=str(err),
                how_to_fix="Provide all required v1 fields and remove unknown keys.",
                where="skills.spec_loader.load_skill_spec",
                fallback_trigger_reason="invalid_skill_spec",
            ),
            validation_error=err,
        ) from err
    except Exception as err:
        raise SkillContractError(
            code=SkillContractErrorCode.INVALID_SPEC,
            payload=SkillContractErrorPayload(
                what_failed="Skill specification parsing failed",
                why=str(err),
                how_to_fix="Ensure the spec is valid JSON/YAML and shaped as an object.",
                where="skills.spec_loader.load_skill_spec",
                fallback_trigger_reason="invalid_skill_spec",
            ),
        ) from err


__all__ = ["load_skill_spec"]
