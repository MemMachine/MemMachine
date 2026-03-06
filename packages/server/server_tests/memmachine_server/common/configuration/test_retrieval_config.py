from __future__ import annotations

from pydantic import SecretStr

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalSkillSessionProvider,
)


def test_retrieval_agent_conf_defaults() -> None:
    conf = RetrievalAgentConf()
    assert conf.skill_session_provider == RetrievalSkillSessionProvider.OPENAI
    assert conf.skill_session_timeout_seconds == 180
    assert conf.skill_session_max_combined_calls == 10
    assert conf.skill_session_log_raw_output is True
    assert conf.openai_native_skill_environment == "local"


def test_retrieval_agent_conf_resolves_anthropic_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-key-from-env")
    conf = RetrievalAgentConf(
        skill_session_provider=RetrievalSkillSessionProvider.ANTHROPIC,
        anthropic_api_key="${TEST_ANTHROPIC_KEY}",
    )

    assert conf.anthropic_api_key == SecretStr("anthropic-key-from-env")
