"""Retrieval-agent configuration models."""

from pydantic import Field

from memmachine_server.common.configuration.mixin_confs import YamlSerializableMixin


class RetrievalAgentConf(YamlSerializableMixin):
    """Configuration for top-level retrieval-agent orchestration."""

    llm_model: str | None = Field(
        default=None,
        description="Default language model used by retrieval-agent strategies (retrieval/planning).",
    )
    answer_llm_model: str | None = Field(
        default=None,
        description="Language model used for answer generation (falls back to llm_model if not set).",
    )
    judge_llm_model: str | None = Field(
        default=None,
        description="Language model used by the LLM judge during evaluation (falls back to llm_model if not set).",
    )
    reranker: str | None = Field(
        default=None,
        description="Default reranker used by retrieval-agent strategies.",
    )
