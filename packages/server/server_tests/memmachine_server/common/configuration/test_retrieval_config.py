from __future__ import annotations

from typing import Any, cast

from pydantic import SecretStr

from memmachine_server.common.configuration.retrieval_config import (
    RetrievalAgentConf,
    RetrievalAgentSessionProvider,
)


def test_retrieval_agent_conf_defaults() -> None:
    conf = RetrievalAgentConf()
    assert conf.agent_session_provider == RetrievalAgentSessionProvider.OPENAI
    assert conf.agent_session_timeout_seconds == 180
    assert conf.agent_session_max_combined_calls == 10
    assert conf.agent_session_log_raw_output is True
    assert conf.openai_native_agent_environment == "local"


def test_retrieval_agent_conf_resolves_anthropic_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "anthropic-key-from-env")
    conf = RetrievalAgentConf(
        agent_session_provider=RetrievalAgentSessionProvider.ANTHROPIC,
        anthropic_api_key=cast(Any, "${TEST_ANTHROPIC_KEY}"),
    )

    assert conf.anthropic_api_key == SecretStr("anthropic-key-from-env")
