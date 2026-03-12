"""Unit tests for installed skill metadata."""

import pytest
from pydantic import ValidationError

from memmachine_client import Skill


def test_skill_fields():
    skill = Skill(
        provider="openai",
        skill_name="retrieve-skill",
        file_ids=("file-1", "file-2"),
        content_hashes=("hash-1", "hash-2"),
    )

    assert skill.provider == "openai"
    assert skill.skill_name == "retrieve-skill"
    assert skill.file_ids == ("file-1", "file-2")
    assert skill.content_hashes == ("hash-1", "hash-2")


def test_skill_frozen():
    skill = Skill(
        provider="anthropic",
        skill_name="retrieve-skill",
        file_ids=("file-1",),
        content_hashes=("hash-1",),
    )

    with pytest.raises(ValidationError, match="frozen"):
        skill.skill_name = "changed"


def test_skill_serialization():
    skill = Skill(
        provider="openai",
        skill_name="retrieve-skill",
        file_ids=("file-1",),
        content_hashes=("hash-1",),
    )

    dumped = skill.model_dump()

    assert dumped == {
        "provider": "openai",
        "skill_name": "retrieve-skill",
        "file_ids": ("file-1",),
        "content_hashes": ("hash-1",),
    }
    assert '"provider":"openai"' in skill.model_dump_json()


def test_skill_rejects_mismatched_parallel_fields():
    with pytest.raises(ValidationError, match="matching lengths"):
        Skill(
            provider="openai",
            skill_name="retrieve-skill",
            file_ids=("file-1", "file-2"),
            content_hashes=("hash-1",),
        )
