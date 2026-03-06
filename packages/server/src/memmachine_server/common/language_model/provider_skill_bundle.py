"""Helpers for provider-native skill bundle attachments."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ProviderSkillBundle(BaseModel):
    """Provider-neutral skill bundle metadata passed to session runtimes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(default="")
    path: str = Field(min_length=1)


def materialize_provider_skill_bundle(
    *,
    name: str,
    description: str,
    skill_markdown: str,
    bundle_root: str | None = None,
) -> ProviderSkillBundle:
    """Materialize markdown skill text as a SKILL.md directory bundle."""
    normalized_name = _normalize_skill_name(name)
    normalized_description = (description or "").strip()
    normalized_markdown = skill_markdown.strip()
    if not normalized_markdown:
        normalized_markdown = normalized_description or normalized_name

    digest = hashlib.sha256(
        f"{normalized_name}\n{normalized_markdown}".encode()
    ).hexdigest()[:16]
    root = (
        Path(bundle_root).expanduser()
        if isinstance(bundle_root, str) and bundle_root.strip()
        else Path(tempfile.gettempdir()) / "memmachine_provider_skills"
    )
    skill_dir = root / f"{normalized_name}-{digest}"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        skill_file.write_text(normalized_markdown, encoding="utf-8")

    return ProviderSkillBundle(
        name=normalized_name,
        description=normalized_description or normalized_name,
        path=str(skill_dir),
    )


def _normalize_skill_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "skill"
