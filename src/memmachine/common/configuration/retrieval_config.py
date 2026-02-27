"""Retrieval-agent configuration models."""

from pydantic import Field

from memmachine.common.configuration.mixin_confs import YamlSerializableMixin


class RetrievalAgentConf(YamlSerializableMixin):
    """Configuration for top-level retrieval-agent orchestration."""

    llm_model: str | None = Field(
        default=None,
        description="Default language model used by retrieval-agent strategies.",
    )
    reranker: str | None = Field(
        default=None,
        description="Default reranker used by retrieval-agent strategies.",
    )
