"""Installed skill metadata for provider-native client execution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class Skill(BaseModel):
    """Immutable metadata for a skill uploaded to a provider Files API."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["anthropic", "openai"]
    skill_name: str = Field(min_length=1)
    file_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_parallel_fields(self) -> Self:
        if not self.file_ids:
            raise ValueError("file_ids must contain at least one uploaded file")
        if len(self.file_ids) != len(self.content_hashes):
            raise ValueError("file_ids and content_hashes must have matching lengths")
        return self
