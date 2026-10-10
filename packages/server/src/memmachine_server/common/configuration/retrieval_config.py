"""Retrieval-agent configuration models."""

from typing import Self

from pydantic import Field, model_validator

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


class ProgressiveRetrievalConf(YamlSerializableMixin):
    """Opt-in limits for progressively widening long-term episodic retrieval."""

    initial_limit: int = Field(default=5, gt=0, strict=True)
    max_candidates: int = Field(default=40, gt=0, strict=True)
    max_rounds: int = Field(default=3, gt=0, strict=True)
    confidence_threshold: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)


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
    progressive: ProgressiveRetrievalConf | None = Field(
        default=None,
        description=(
            "When set, use bounded progressive retrieval for agent-mode long-term "
            "episodic search instead of tool selection. Omit to keep existing behavior."
        ),
    )

    @model_validator(mode="after")
    def validate_retrieval_strategy(self) -> Self:
        """Reject ambiguous selection of two opt-in strategies."""
        if self.progressive is not None and self.use_optimized_coq:
            raise ValueError("progressive and use_optimized_coq cannot both be enabled")
        return self
