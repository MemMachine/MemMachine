"""Factory helpers for retrieval-skill construction."""

from memmachine_server.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine_server.common.language_model import (
    LanguageModel,
    SkillLanguageModel,
    SkillSessionModelProtocol,
    create_skill_session_model,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.retrieval_skill.common.skill_api import (
    SkillToolBase,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.skills.retrieve_skill import RetrieveSkill
from memmachine_server.retrieval_skill.subskills import MemMachineSkill


def create_retrieval_skill(
    *,
    model: LanguageModel,
    reranker: Reranker,
    retrieval_conf: RetrievalAgentConf | None = None,
    skill_name: str = "RetrieveSkill",
    skill_session_model: SkillSessionModelProtocol | None = None,
) -> SkillToolBase:
    """Create the top-level retrieval skill orchestrator."""
    if skill_name != "RetrieveSkill":
        raise ValueError(
            "create_retrieval_skill only supports skill_name='RetrieveSkill'. "
            f"Received: {skill_name!r}"
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
        if retrieval_conf is not None:
            resolved_skill_session_model = create_skill_session_model(
                model=model,
                retrieval_conf=retrieval_conf,
            )
        else:
            resolved_skill_session_model = (
                SkillLanguageModel.from_openai_responses_language_model(model)
            )

    retrieve_extra_params: dict[str, object] = {
        "fallback_tool_name": memory_skill.skill_name,
        "skill_session_model": resolved_skill_session_model,
    }
    if retrieval_conf is not None:
        retrieve_extra_params["global_timeout_seconds"] = (
            retrieval_conf.skill_session_timeout_seconds
        )
        retrieve_extra_params["max_combined_calls"] = (
            retrieval_conf.skill_session_max_combined_calls
        )
        if retrieval_conf.skill_native_bundle_root:
            retrieve_extra_params["native_skill_bundle_root"] = (
                retrieval_conf.skill_native_bundle_root
            )

    retrieve_skill = RetrieveSkill(
        SkillToolBaseParam(
            model=model,
            children_tools=[memory_skill],
            extra_params=retrieve_extra_params,
            reranker=reranker,
        )
    )
    return retrieve_skill
