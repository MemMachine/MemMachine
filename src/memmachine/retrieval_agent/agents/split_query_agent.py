"""Query-splitting agent for independent single-hop sub-queries."""

import asyncio
import copy
import logging
import time
from typing import Any, cast

from memmachine.common.language_model.language_model import LanguageModel
from memmachine.episodic_memory.declarative_memory import Episode
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)

logger = logging.getLogger(__name__)

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
SPLIT_QUERY_PROMPT = """You are a search expert. Transform the input query into either:
- multiple single-hop sub-queries (2-6 lines), or
- the original query unchanged (1 line),
following the rules below.

Query
{query}

Mechanism (what to do, then how to do it)

1) Decide whether to split (default: do NOT split)
- Do NOT split if the answer can be retrieved from a single page/infobox/database record/field set for the same entity and timeframe (i.e., one co-located source would likely co
ntain it).
- Split ONLY if you must retrieve >=2 distinct facts that are not co-located OR are for different entities and/or different timepoints/locations/contexts.
- Tie-breaker: when unsure, prefer NOT splitting.

2) Special cases that limit splitting
- Multi-constraint single-entity queries (e.g., "A's birth date and birthplace"):
  - Keep as ONE query if those attributes are typically co-located in the same reference entry for A.
  - Split only if the attributes are typically separate lookups (or clearly not co-located).
- List-style questions ("all/each/every ... and their ..."):
  - Keep as-is unless the query explicitly names 2-6 specific entities/subjects that can each become one sub-query.
  - Do not split if it would likely require more than 6 lines.

3) If splitting: produce single-hop fact-lookups only
- Each sub-query must be directly answerable by one fact lookup (one field/value) with no derived operations.
- Explicit ban: do NOT use derived/operation wording in sub-queries, including (non-exhaustive) "compare," "difference," "between," "rate," "top," "average," "change," "increase
," "decrease," "percent," "rank," "versus," "more/less than."
- Rewrite derived intents into pure fact retrievals (e.g., "between X and Y" -> ask for X and ask for Y; "compare A and B in 2023" -> ask A in 2023 and ask B in 2023).

4) Preserve intent and attach constraints correctly
- Keep the same entities/aliases and the same constraints (timeframe, location, context, units) from the original query.
- For mixed structures, maintain left-to-right entity/subject order AND attach any paired constraints to every relevant sub-query (e.g., "in 2023," "in Paris," "during WWII" app
ears on each line it applies to).
- Do not add assumptions or extra constraints.

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

Conjunction + timeframe (splittable)
Query: What were the populations of Canada and Mexico in 2021?
Output:
What was the population of Canada in 2021?
What was the population of Mexico in 2021?

"Between" question (rewrite into facts)
Query: How many days are there between Tom's birthday and Mike's birthday?
Output:
What is Tom's birthday?
What is Mike's birthday?

Relational (not splittable if single lookup)
Query: Who is Taylor Swift's boyfriend?
Output:
Who is Taylor Swift's boyfriend?

Pronoun resolution (splittable)
Query: What country is he the president of in 2024?
Output:
Who does "he" refer to in the context of "the president of in 2024"?
What country is [resolved person] the president of in 2024?

Multi-entity multi-attribute with constraints (splittable)
Query: What were Japan's GDP in 2023 and Germany's GDP in 2023?
Output:
What was the GDP of Japan in 2023?
What was the GDP of Germany in 2023?

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

class SplitQueryAgent(AgentToolBase):
    """Split complex queries into single-hop sub-queries when beneficial."""

    def __init__(self, param: AgentToolBaseParam) -> None:
        """Initialize with a language model and split prompt."""
        super().__init__(param)
        if self._model is None:
            raise ValueError("Model is not set")
        self._prompt = (param.extra_params or {}).get(
            "split_prompt",
            SPLIT_QUERY_PROMPT,
        )

    @property
    def agent_name(self) -> str:
        return "SplitQueryAgent"

    @property
    def agent_description(self) -> str:
        return """This agent splits a complex query into a few simple sub queries and perform the query on each sub query. It aggregates
        the result from each sub query and return the final result.
        """

    @property
    def accuracy_score(self) -> int:
        return 6

    @property
    def token_cost(self) -> int:
        return 3

    @property
    def time_cost(self) -> int:
        return 5

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        logger.info("CALLING %s with query: %s", self.agent_name, query.query)
        perf_matrics: dict[str, Any] = {
            "queries": [],
            "llm_time": 0.0,
            "agent": self.agent_name,
        }
        prompt = self._prompt.format(query=query.query)
        llm_start = time.time()
        rsp, _, input_token, output_token = await cast(LanguageModel, self._model).generate_response_with_token_usage(user_prompt=prompt)
        perf_matrics["llm_time"] += time.time() - llm_start
        sub_queries: list[str] = []
        for line in rsp.split("\n"):
            if line.strip() == "":
                continue
            sub_queries.append(line.strip())
        if len(sub_queries) == 0:
            sub_queries = [query.query]

        result: list[Episode] = []
        tasks = []
        for sub_query in sub_queries:
            perf_matrics["queries"].append(sub_query)
            param = copy.deepcopy(query)
            param.query = sub_query
            # TODO: make this self-adaptive
            # param.limit /= 2
            tasks.append(super().do_query(policy, param))
        results = await asyncio.gather(*tasks)
        for res, perf in results:
            if res is None:
                continue
            result.extend(res)
            perf_matrics = self._update_perf_matrics(perf, perf_matrics)

        self._update_perf_matrics(
            {
                "input_token": input_token,
                "output_token": output_token,
            },
            perf_matrics,
        )

        # Rerank base on all queries concatenated
        param = copy.deepcopy(query)
        if len(sub_queries) > 1:
            param.query += "\n".join(sub_queries)
        final_episodes = await self._do_rerank(param, result)
        return final_episodes, perf_matrics
