"""Retrieval-agent configuration models."""

from pydantic import Field

from memmachine_server.common.configuration.mixin_confs import YamlSerializableMixin


class OptimizedCoqConf(YamlSerializableMixin):
    """Settings for RaragQueryAgent (optimized ChainOfQueryAgent variant)."""

    multi_hop_decomposer: bool = Field(
        default=False,
        description=(
            "When true, RaragQueryAgent splits multi-hop queries with the "
            "embedded non-LLM decomposer (spaCy-based). When false or unset, "
            "the original LLM-based hop splitting is used."
        ),
    )
    multi_hop_sub_limit: int = Field(
        default=20,
        description=(
            "Fixed per-sub-search limit used by RaragQueryAgent for the A, "
            "A->C, and combined-query searches. Independent of the user-"
            "configured top-k limit, which only governs the final return cap."
        ),
    )


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
    use_optimized_coq: bool = Field(
        default=False,
        description=(
            "When true, the ChainOfQueryAgent slot is filled by RaragQueryAgent, "
            "an optimized multi-hop retrieval variant. When false or unset, the "
            "original ChainOfQueryAgent is used."
        ),
    )
    optimized_coq: OptimizedCoqConf | None = Field(
        default=None,
        description="RaragQueryAgent (optimized ChainOfQueryAgent) settings.",
    )
