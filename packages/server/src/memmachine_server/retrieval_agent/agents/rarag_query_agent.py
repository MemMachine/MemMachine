"""RaragQueryAgent: optimized ChainOfQueryAgent for multi-hop retrieval.

Decomposes multi-hop queries into hops (non-LLM decomposer or LLM-based
fallback), searches A and A->C in parallel, finds overlapping memories, builds
combined queries from the top overlaps, deduplicates, and returns up to 200
episodes. Positioned as an optimized drop-in for the ChainOfQueryAgent slot.
"""

import asyncio
import logging
import time
from typing import Any, cast

from memmachine_server.common.episode_store import Episode
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

logger = logging.getLogger(__name__)

# Non-LLM hop decomposer (embedded helper for RaragQueryAgent).
try:
    from memmachine_server.retrieval_agent.agents.decomposer.decomposer import (
        MultiHopDecomposer,
    )
    DECOMPOSER_AVAILABLE = True
    logger.info("Decomposer imported successfully!")
except ImportError as e:
    DECOMPOSER_AVAILABLE = False
    MultiHopDecomposer = None
    logger.error("Decomposer import FAILED: %s", e)

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
RARAG_QUERY_PROMPT = """You are a search expert. Transform the input query into either:
- multiple single-hop sub-queries (2-6 lines), or
- the original query unchanged (1 line),
following the rules below.

Query
{query}

Mechanism (what to do, then how to do it)

1) Decide whether to split (default: Always SPLIT)
- Always attempt to split the query into single-hop sub-queries.
- Split if the query requires >=2 distinct facts that are not co-located OR are for different entities and/or different timepoints/locations/contexts.
- Multi-hop questions (e.g., "Why did X's Y...", "What is A's B's C?") MUST be split into separate single-hop queries.
- Tie-breaker: when unsure, prefer SPLITTING.

2) Special cases for multi-hop splitting
- Multi-hop questions (e.g., "Why did the founder of Versus die?"):
  - Split into: "Who founded Versus?" + "Why did [founder] die?"
  - Each nested reference = one hop = one sub-query.
- Nested references (e.g., "director of film X", "spouse of the CEO"):
  - First sub-query: resolve the reference (e.g., "Who directed film X?")
  - Second sub-query: use the resolved entity for the next fact.
- List-style questions ("all/each/every ... and their ..."):
  - Split if the query explicitly names 2-6 specific entities/subjects.
  - Do not split if it would likely require more than 6 lines.

3) Split with minimal modification (use pronouns when possible)
- For multi-hop questions, split into: (1) the nested reference phrase, (2) the main question with minimal pronoun substitution.
- Replace the nested reference with a minimal pronoun/reference (e.g., "he", "she", "it", "they", "this person", "this place") in the second part.
- Keep the rest of the original question wording exactly as-is.
- Example: "Why did the founder of Versus die?" → "the founder of Versus" + "Why did he die?"
- Each sub-query will be used for similarity search as-is.

4) Preserve original wording with minimal substitution
- Keep the original question structure and wording as much as possible.
- Only replace the nested reference with a minimal pronoun/reference (e.g., "he", "she", "it", "they", "this person", "this place", "the company").
- Do NOT add extra words, explanations, or hints beyond the minimal pronoun substitution.
- Do NOT use placeholders like "[founder]" or "[resolved person]".

5) Handling common structures
- Conjunctions ("A and B"): one sub-query per entity/subject for the same attribute and constraints.
- Multi-entity multi-attribute: split by entity first in left-to-right order; within each entity, include only the minimal single-hop attribute per line needed to cover the orig
inal query (while respecting the 2-6 line cap).
- Relational questions ("A's relationship to B"):
  - Keep as one query if a single fact lookup answers it (e.g., "Who is A's spouse?").
  - Only add identity-resolution sub-queries if necessary to retrieve the relationship (see pronouns/ambiguity rule).
- Pronouns / ambiguous references:
  - If a pronoun exists AND its referent is not explicitly stated in the query, first add exactly one sub-query:
    Who does "[pronoun]" refer to in the context of "[minimal relevant context from the query]"?
  - Then add only the needed fact-lookup sub-queries.
  - If the referent is explicitly stated, do not add a resolution query.

6) Internal duplicate guardrail (must pass)
- Ensure no two lines ask for the same attribute of the same entity under the same timeframe/location/context.

Examples

Conjunction + timeframe (splittable, extract phrases)
Query: What were the populations of Canada and Mexico in 2021?
Output:
the population of Canada in 2021
the population of Mexico in 2021

"Between" question (extract facts only)
Query: How many days are there between Tom's birthday and Mike's birthday?
Output:
Tom's birthday
Mike's birthday

Relational (not splittable if single lookup)
Query: Who is Taylor Swift's boyfriend?
Output:
Who is Taylor Swift's boyfriend?

Pronoun resolution (splittable, extract phrase only)
Query: What country is he the president of in 2024?
Output:
Who does "he" refer to in the context of "the president of in 2024"?
What country is "he" the president of in 2024?

Multi-entity multi-attribute with constraints (splittable, extract phrases)
Query: What were Japan's GDP in 2023 and Germany's GDP in 2023?
Output:
Japan's GDP in 2023
Germany's GDP in 2023

Multi-hop (splittable, minimal pronoun substitution)
Query: Why did the founder of Versus die?
Output:
the founder of Versus
Why did he die?

Multi-hop with nested reference (splittable, minimal substitution)
Query: What is the birthplace of the director of film Beat Girl?
Output:
the director of film Beat Girl
What is the birthplace of he/she?

Multi-hop chain (splittable, minimal pronoun)
Query: What year was the wife of the governor of California born?
Output:
the wife of the governor of California
What year was she born?

List-style (do not split)
Query: List all members of the United Nations and their admission years.
Output:
List all members of the United Nations and their admission years?

Output Format (strict)
- Output ONLY the resulting queries.
- 1-6 lines total (if split: 2-6 lines; if not: 1 line).
- One query per line.
- Each line must be a full question ending with "?".
- No numbering, bullets, quotes, headings, or extra text.
- No blank lines.
- Final self-check before output:
  - Line count is valid (1-6; if split then 2-6).
  - Every line ends with "?".
  - No derived/operation wording appears in any sub-query.
  - No duplicate attribute/entity/timeframe queries.
"""


class RaragQueryAgent(AgentToolBase):
    """Optimized ChainOfQueryAgent for multi-hop retrieval (RARAG)."""

    def __init__(self, param: AgentToolBaseParam) -> None:
        """Initialize with a language model and split prompt."""
        super().__init__(param)
        if self._model is None:
            raise ValueError("Model is not set")
        self._prompt = (param.extra_params or {}).get(
            "split_prompt",
            RARAG_QUERY_PROMPT,
        )
        # Hop-splitting strategy: non-LLM decomposer when configured and
        # available, otherwise the original LLM-based splitting.
        self._use_decomposer = bool(
            (param.extra_params or {}).get("multi_hop_decomposer", False)
        ) and DECOMPOSER_AVAILABLE
        self._decomposer = MultiHopDecomposer() if DECOMPOSER_AVAILABLE else None
        # Fixed per-sub-search limit for A, A->C, and combined-query searches,
        # independent of the user-configured top-k limit (query.limit).
        self._sub_search_limit = int(
            (param.extra_params or {}).get("multi_hop_sub_limit", 20)
        )
        logger.info(
            "RaragQueryAgent hop splitting: %s (sub_limit=%d)",
            "non-LLM decomposer"
            if self._use_decomposer
            else "LLM-based",
            self._sub_search_limit,
        )

    @property
    def agent_name(self) -> str:
        # Keep "ChainOfQueryAgent" so this optimized variant drops into the
        # ChainOfQueryAgent slot of ToolSelectAgent when use_optimized_coq is on.
        return "ChainOfQueryAgent"

    @property
    def agent_description(self) -> str:
        return (
            "Optimized ChainOfQueryAgent (RARAG): decomposes multi-hop queries "
            "and searches overlap-driven combined queries."
        )

    @property
    def accuracy_score(self) -> int:
        return 6

    @property
    def token_cost(self) -> int:
        return 3

    @property
    def time_cost(self) -> int:
        return 5

    async def _split_with_llm(
        self,
        query: QueryParam,
    ) -> tuple[list[str], int, int, float]:
        """Split the query into sub-queries using the LLM.

        Returns (sub_queries, input_token, output_token, llm_time).
        """
        prompt = self._prompt.format(query=query.query)
        llm_start = time.time()
        rsp, _, input_token, output_token = await cast(
            LanguageModel, self._model
        ).generate_response_with_token_usage(user_prompt=prompt)
        llm_time = time.time() - llm_start
        sub_queries: list[str] = []
        for line in rsp.split("\n"):
            if line.strip() == "":
                continue
            sub_queries.append(line.strip())
        if len(sub_queries) == 0:
            sub_queries = [query.query]
        logger.info("Step 0: Split using LLM into %d sub-queries", len(sub_queries))
        return sub_queries, input_token, output_token, llm_time

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        # Truncate query for logging to avoid excessive output
        query_preview = query.query[:200] + "..." if len(query.query) > 200 else query.query
        logger.debug("CALLING %s with query: %s", self.agent_name, query_preview)
        perf_metrics: dict[str, Any] = {
            "queries": [],
            "llm_time": 0.0,
            "agent": self.agent_name,
        }

        # Hop splitting: non-LLM decomposer when configured, else LLM-based.
        # If the decomposer cannot handle the query (not multi-hop per its
        # rules, or it raises), fall back to LLM-based splitting so the agent
        # still gets a real hop decomposition.
        sub_queries: list[str] = []
        input_token = 0
        output_token = 0

        use_decomposer_result = False
        if self._use_decomposer:
            try:
                result = self._decomposer.decompose(query.query)
                if result.is_multi_hop:
                    sub_queries = [result.first_hop, result.second_hop_template.replace("[HOP]", result.first_hop)]
                    logger.info("Step 0: Decomposed using non-LLM decomposer: %s -> %s", result.first_hop, result.second_hop_template)
                    use_decomposer_result = True
                else:
                    logger.info("Step 0: Decomposer says not multi-hop, falling back to LLM")
            except Exception as e:
                logger.warning("Step 0: Decomposer failed, falling back to LLM: %s", e)

        if not use_decomposer_result:
            sub_queries, input_token, output_token, llm_time = await self._split_with_llm(query)
            perf_metrics["llm_time"] += llm_time

        # Strategy:
        # 1. Search with A (first sub-query) and A->C (original query) in parallel
        # 2. Find overlapping episodes between A and A->C by UID
        # 3. For each overlap, build a combined query (overlap content + C) and search
        # 4. Deduplicate combined results by UID and return them.
        #    A and A->C results are used only to find overlaps and are NOT returned.

        combined_query_results: list[Episode] = []

        # Step 1: Search with A (first sub-query) and A->C (original query) in parallel.
        #
        # Rarag is intentionally a 2-hop design: the A / A->C overlap + B'->C
        # mechanism is built around exactly two hops. Only the first two
        # sub-queries are consumed here -- A is sub_queries[0] (the first hop),
        # and the original question serves as the second hop (A->C). The LLM
        # splitter may return more than two sub-queries, but any beyond the
        # first two are intentionally ignored; Rarag is not meant to handle
        # 3+ hop questions on its own.
        query_a = sub_queries[0] if len(sub_queries) >= 1 else query.query
        query_c = sub_queries[1] if len(sub_queries) >= 2 else None
        query_original = query.query

        logger.info("Step 1: Searching with A='%s' and A->C='%s' (in parallel)", query_a, query_original)

        # Search with A and A->C in parallel
        perf_metrics["queries"].append(query_a)
        perf_metrics["queries"].append(query_original)

        param_a = query.model_copy()
        param_a.query = query_a
        param_a.limit = self._sub_search_limit
        param_original = query.model_copy()
        param_original.query = query_original
        param_original.limit = self._sub_search_limit

        # asyncio.gather returns list of results, each result is (episodes, perf_metrics) tuple
        results = await asyncio.gather(
            super().do_query(policy, param_a),
            super().do_query(policy, param_original)
        )
        episodes_a, perf_a = results[0]
        episodes_original, perf_original = results[1]

        # Update perf metrics
        perf_metrics = self._update_perf_metrics(perf_a, perf_metrics)
        perf_metrics = self._update_perf_metrics(perf_original, perf_metrics)

        logger.info("Step 1: A=%d episodes, A->C=%d episodes (parallel search complete)", len(episodes_a), len(episodes_original))

        # Step 2: Find overlapping episodes between A and A->C by UID
        # Build a set of UIDs from episodes_original for O(1) lookup
        episodes_original_uids = {ep.uid for ep in episodes_original}
        # Preserve order from episodes_a (instead of using set intersection which is non-deterministic)
        overlap_uids = []
        for ep in episodes_a:
            if ep.uid in episodes_original_uids:
                overlap_uids.append(ep.uid)

        logger.info("Step 2: Found %d overlapping UIDs between A and A->C", len(overlap_uids))

        # Step 3: For each overlapping episode (by UID), create combined query:
        # overlap_content + C, and search sequentially.
        # Sequential, not parallel: concurrent reranker calls raise per-search
        # latency (server contention) enough to offset the concurrency gain.
        if query_c and overlap_uids:
            # Map UID -> episode for episodes_a (combined-query content source).
            episodes_a_by_uid = {ep.uid: ep for ep in episodes_a}
            # Overlap cap = 400 / sub_search_limit: keeps raw results (overlap *
            # sub_search_limit) near 400, so ~half dedup lands near the 200 cap.
            # episodes_a is in reranker-score order, so these are the top overlaps.
            overlap_limit = (
                int(200 / (self._sub_search_limit / 2))
                if self._sub_search_limit > 0
                else 0
            )
            overlap_uids_limited = overlap_uids[:overlap_limit]

            logger.info("Step 3: Searching with %d combined queries (cap=%d, total overlaps=%d, sequentially)", len(overlap_uids_limited), overlap_limit, len(overlap_uids))
            for i, overlap_uid in enumerate(overlap_uids_limited):
                overlap_content = episodes_a_by_uid[overlap_uid].content
                combined_query = f"{overlap_content} {query_c}"
                logger.info("Step 3.%d: Combined query (len=%d): '%s'", i+1, len(combined_query), combined_query[:100] + "..." if len(combined_query) > 100 else combined_query)

                perf_metrics["queries"].append(combined_query)
                param_combined = query.model_copy()
                param_combined.query = combined_query
                param_combined.limit = self._sub_search_limit
                episodes_combined, perf_combined = await super().do_query(policy, param_combined)
                perf_metrics = self._update_perf_metrics(perf_combined, perf_metrics)
                combined_query_results.extend(episodes_combined)
        else:
            logger.info("Step 3: Skipping combined query search (no C query or no overlap)")

        self._update_perf_metrics(
            {
                "input_token": input_token,
                "output_token": output_token,
            },
            perf_metrics,
        )

        # Step 5: Deduplicate combined query results by UID
        logger.info("Step 5: Deduplicating %d combined query results by UID", len(combined_query_results))

        # Fallback: if there were no overlaps (no combined queries), return
        # the A->C (original query) results instead of an empty set, so a
        # zero-overlap multi-hop question still yields memories.
        dedup_source = combined_query_results if combined_query_results else episodes_original

        # Deduplicate by episode UID (keep first occurrence by created_at)
        seen_uids = set()
        deduplicated = []
        for ep in sorted(dedup_source, key=lambda x: x.created_at):  # Sort by created_at first
            if ep.uid not in seen_uids:
                seen_uids.add(ep.uid)
                deduplicated.append(ep)

        logger.info("Step 5: After deduplication, %d episodes remain", len(deduplicated))

        # Step 6: Return the deduplicated combined results capped at 200. No
        # 2nd-stage reranking. The overlap cap (Step 3) is sized so that, after
        # ~half dedup, the result lands near this fixed 200 cap.
        final_episodes = deduplicated[:200]
        logger.info("Step 6: Returning %d episodes (cap=200, dedup=%d)", len(final_episodes), len(deduplicated))
        return final_episodes, perf_metrics
