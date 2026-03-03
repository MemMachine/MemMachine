"""Shared interfaces and base types for retrieval skills."""

from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBase,
    SkillToolBaseParam,
)

__all__ = [
    "QueryParam",
    "QueryPolicy",
    "SkillToolBase",
    "SkillToolBaseParam",
]
