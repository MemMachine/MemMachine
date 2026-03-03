"""Factory helpers for retrieval-skill construction."""

import logging

from memmachine_server.common.language_model import (
    LanguageModel,
    SkillLanguageModel,
    SkillSessionModelProtocol,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.retrieval_skill.common.skill_api import (
    SkillToolBase,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.skills.retrieve_skill import RetrieveSkill
from memmachine_server.retrieval_skill.subskills import MemMachineSkill

logger = logging.getLogger(__name__)


def create_retrieval_skill(
    *,
    model: LanguageModel,
    reranker: Reranker,
    skill_name: str = "RetrieveSkill",
    skill_session_model: SkillSessionModelProtocol | None = None,
) -> SkillToolBase:
    """Create the top-level retrieval skill orchestrator."""
    if skill_name != "RetrieveSkill":
        logger.warning(
            "Ignoring legacy retrieval skill_name '%s'; using RetrieveSkill.",
            skill_name,
        )

    memory_skill = MemMachineSkill(
        SkillToolBaseParam(
            model=None,
            children_tools=[],
            extra_params={},
            reranker=reranker,
        ),
    )

    resolved_skill_session_model = skill_session_model
    if resolved_skill_session_model is None:
        resolved_skill_session_model = (
            SkillLanguageModel.from_openai_responses_language_model(model)
        )

    retrieve_skill = RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[memory_skill],
            extra_params={
                "fallback_tool_name": memory_skill.skill_name,
                "skill_session_model": resolved_skill_session_model,
            },
            reranker=reranker,
        )
    )
    return retrieve_skill
