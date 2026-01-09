"""Embedder-based reranker implementation."""

from pydantic import BaseModel, Field, InstanceOf, TypeAdapter

from memmachine.common.language_model import LanguageModel

from .reranker import Reranker

RERANKING_PROMPT_TEMPLATE = (
    "You are given a mapping of ids to candidates and a query. "
    "Your task is to rank the candidate ids in an array that you create, "
    "so that the query can be answered using as few candidates as possible starting from the first element in the array.\n"
    "\n"
    "<instructions>\n"
    "1. Prioritize the ability to answer the query with the fewest candidates from the start of the array above all other considerations.\n"
    "2. The candidates are initially ordered by retrieval cue association.\n"
    "3. Your array must contain all candidate ids exactly once.\n"
    "</instructions>\n"
    "\n"
    "<candidates>\n"
    "{candidates_mapping}\n"
    "</candidates>\n"
    "\n"
    "<query>\n"
    "{query}\n"
    "</query>"
)


class LanguageModelRerankerParams(BaseModel):
    """Parameters for LanguageModelReranker."""

    language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description="An instance of a LanguageModel to use for scoring candidates",
    )
    max_num_candidates: int = Field(
        50,
        description="Maximum number of candidates to score",
    )


class LanguageModelReranker(Reranker):
    """Reranker that uses a generative language model to score candidate relevance."""

    def __init__(self, params: LanguageModelRerankerParams) -> None:
        """Initialize a LanguageModelReranker with the provided configuration."""
        super().__init__()

        self._language_model = params.language_model
        self._max_num_candidates = params.max_num_candidates

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using a language model."""
        if len(candidates) == 0:
            return []

        scored_candidates = candidates[: self._max_num_candidates]

        scored_candidates_mapping = {
            f"id{index}": candidate for index, candidate in enumerate(scored_candidates)
        }

        reranking_prompt = RERANKING_PROMPT_TEMPLATE.format(
            candidates_mapping=scored_candidates_mapping,
            query=query,
        )

        ranked_candidate_ids = []
        while len(ranked_candidate_ids) < len(scored_candidates):
            response = await self._language_model.generate_parsed_response(
                output_format=RerankResponse,
                user_prompt=reranking_prompt,
            )

            if response is None:
                continue

            validated_response = TypeAdapter(RerankResponse).validate_python(
                response,
            )

            ranked_candidate_ids = validated_response.ordered_candidate_ids

        scored_candidate_ranks = {
            candidate_id: rank for rank, candidate_id in enumerate(ranked_candidate_ids)
        }

        return [
            -float(scored_candidate_ranks.get(f"id{index}", "inf"))
            for index in range(len(candidates))
        ]


class RerankResponse(BaseModel):
    """Response containing an array of candidate ids in ranked order."""

    ordered_candidate_ids: list[str] = Field(default_factory=list)
