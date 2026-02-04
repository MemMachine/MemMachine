import copy
import logging
import json
import time
from typing import Any, cast
from collections.abc import Iterable

from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryPolicy,
    QueryParam,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.episodic_memory.declarative_memory import (
    Episode,
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

logger = logging.getLogger(__name__)

# Latest optimized version
COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT = """You are a meticulous expert in retrieval-augmented question answering evaluation and query rewriting.

Task: Given (1) an original user query, (2) rewritten queries already tried, and (3) retrieved documents, you must decide whether the documents are sufficient to answer the orig
inal query directly, completely, and explicitly. If insufficient, generate the NEXT BEST rewritten subquery to retrieve the missing evidence. If sufficient, set the rewritten qu
ery to the original query.

Hard constraints:
- Use ONLY the provided retrieved documents for sufficiency judgment. Do NOT use external knowledge, browsing, assumptions, or plausibility.
- Do NOT invent new entities. Only use entity names/terms present in the retrieved documents and/or original query.
- Output ONLY a valid JSON object with EXACTLY these keys and no others:
  - "is_sufficient" (boolean)
  - "evidence_indices" (list of 0-based integers)
  - "new_query" (string, single-line)
  - "confidence_score" (number 0..1)

Inputs (schemas):
- Original Query: a string.
- Rewritten Queries Tried: {used_query}
  - May be: (a) an array/list of strings, (b) a newline-separated string, or (c) empty/null.
  - Treat it as the full set of prior attempted rewritten queries.
- Retrieved Documents: {retrieved_episodes}
  - Must be treated as an ordered list/array of documents.
  - Each document may be a string or an object; regardless, assume each has textual content you can read.
  - evidence_indices refer to this list order (0-based).
  - If empty/null/unreadable: treat as no evidence.

Decision procedure (mechanism-first):
1) Normalize inputs (internal):
- Parse used_query into a list of strings if needed; if empty/null, use [].
- Treat retrieved_episodes as a list; if empty/null, use [].

2) Decompose the Original Query (internal):
Identify ALL required informational components needed to answer every part of the query:
- Key entities (people/organizations/places/products)
- Required attributes (names/dates/locations/numbers/definitions/specs)
- Required relationships / multi-hop chains (each hop/link)
- Constraints (time range, “latest”, comparisons, “list all”, counts, completeness scope)

3) Evidence scan and relevance:
- A document is “relevant” ONLY if it explicitly contains at least one required fact OR explicitly establishes an intermediate link in a required multi-hop chain.
- Collect evidence_indices as the set of all relevant documents that contribute required facts/links.
- If no document contributes any required fact/link, evidence_indices must be [].

4) Sufficiency standard (strict):
Set is_sufficient = true ONLY if:
- The retrieved documents explicitly contain all facts needed to answer every component of the original query, AND
- Any required multi-hop chain has EVERY link explicitly supported in the documents, AND
- Any requirement for exact details (names/dates/locations/numbers/specs) is explicitly present, AND
- Any “how many / list all / compare / full coverage” requirement is satisfiable from documents that clearly cover the complete scope.
Otherwise set is_sufficient = false.
If uncertain at any point, choose is_sufficient = false.

NEXT BEST query objective (only when is_sufficient=false):
Generate ONE single-line new_query that maximizes the chance of retrieving the missing evidence, using this ranking priority:
1) Earliest blocking hop: Target the first missing link that prevents completion of the query’s required chain(s).
2) Specificity: Use the most specific entity names and relation terms available from the retrieved documents (and original query) to reduce ambiguity.
3) Minimality: Ask for exactly the missing fact/link (not the whole original question).
4) Novelty vs tried queries: Avoid repeating prior rewritten queries.

Missing-evidence identification (internal only; do NOT output):
- Determine the minimal missing fact(s) that, if retrieved, would allow answering using (retrieved documents + missing evidence).
- Prefer missing evidence framed as a single subject-focused fact (e.g., “X’s manager”, “Y’s birthplace”, “Z battery capacity”).

Rewritten query generation rules:
- If is_sufficient = true:
  - new_query MUST equal the original query exactly.
- If is_sufficient = false:
  - new_query MUST be a single-line question/phrase that targets the missing evidence per the NEXT BEST objective.
  - Must not introduce new entities not present in retrieved documents/original query.
  - Must avoid duplication of previously tried queries:
    - Normalize by lowercasing and collapsing internal whitespace.
    - If used_query contains multiple items, compare against all of them.
    - If the best candidate matches a tried query after normalization, rephrase with the same intent (synonyms/reordering) while staying equally specific.
  - If you cannot produce a better query than what was tried (or no grounded entities exist to target), set new_query to the original query exactly.

Confidence score calibration:
- confidence_score reflects certainty in your is_sufficient decision (not answer correctness).
- Use these anchors:
  - 0.90–1.00: Very clear sufficiency/insufficiency with explicit supporting/absent facts.
  - 0.60–0.89: Moderate clarity; some ambiguity but decision is still well-supported.
  - 0.30–0.59: Low clarity; documents are noisy/partial; you still must err insufficient.
  - 0.00–0.29: Extremely unclear; empty/unreadable docs or severe mismatch.
- If you chose is_sufficient=false due to uncertainty, keep confidence_score below 0.70.

Edge cases:
- If retrieved_episodes is empty or contains no relevant facts: is_sufficient=false, evidence_indices=[], and produce the most targeted new_query you can grounded in the origina
l query (without inventing new entities).
- If the original query is underspecified/ambiguous and documents do not resolve it explicitly: is_sufficient=false.

Now perform the task using:
**Original Query**
{original_query}

**Rewritten Queries Tried**
{used_query}

**Retrieved Documents**
{retrieved_episodes}

Output the JSON object only.
"""

class ChainOfQueryAgent(AgentToolBase):
    def __init__(self, param: AgentToolBaseParam):
        super().__init__(param)
        extra_params = param.extra_params or {}
        self._combined_prompt: str = extra_params.get("combined_prompt", COMBINED_SUFFICIENCY_AND_REWRITE_PROMPT)
        self._max_attempts: int = extra_params.get("max_attempts", 3)
        self._confidence_score: float = extra_params.get("confidence_score", 0.8)
        if self._model is None:
            raise ValueError("Model is not set")

    @property
    def agent_name(self) -> str:
        return "ChainOfQueryAgent"

    @property
    def agent_description(self) -> str:
        return description
    
    @property
    def accuracy_score(self) -> int:
        return 10

    @property
    def token_cost(self) -> int:
        return 9

    @property
    def time_cost(self) -> int:
        return 10

    def _last_brace_block(self, text: str) -> str | None:
        end = text.rfind("}")
        if end == -1:
            return None

        depth = 0
        for i in range(end, -1, -1):
            ch = text[i]
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    return text[i:end + 1]
        return None

    def _init_perf_matrics(self) -> dict[str, Any]:
        return {
            "queries": [],
            "is_sufficient": [],
            "evidence": [],
            "confidence_scores": [],
            "memory_retrieval_time": 0.0,
            "llm_time": 0.0,
            "input_token": 0,
            "output_token": 0,
        }

    async def combined_check_and_rewrite(
        self,
        query: QueryParam,
        retrieved_episodes: Iterable[Episode],
        retrived_evidence: Iterable[Episode],
        used_queries: list[str],
    ) -> dict[str, Any]:
        context: str = ""
        evidence = set(retrived_evidence)
        episodes = sorted(
            set(retrieved_episodes).union(retrived_evidence),
            key=lambda e: (e.timestamp is None,
                        e.timestamp)
        )
        idx = 0
        for episode in episodes:
            context += f"[{idx}] {DeclarativeMemory.string_from_episode_context([episode])}"
            idx += 1
        used_query_str = "\n".join(used_queries)
        prompt = self._combined_prompt.format(
            original_query=query.query,
            used_query=used_query_str,
            retrieved_episodes=context
        )
        m = cast(LanguageModel, self._model)
        rsp, _, input_token, output_token = await m.generate_response_with_token_usage(user_prompt=prompt)
        logger.debug(f"Combined Check and Rewrite: {rsp}")
        json_parsable_str = ""
        response = {}
        retry = True
        while retry:
            try:
                # Get last {} JSON block
                json_parsable_str = self._last_brace_block(rsp)
                response = json.loads(json_parsable_str)
                break
            except Exception as e:
                print(f"WARNING: Failed to parse combined check and rewrite response JSON: {rsp}\nFinal string used: {json_parsable_str} error: {e}")
                if retry == False:
                    break
                retry = False
                continue
        for idx_val in response.get("evidence_indices", []):
            if idx_val < 0 or idx_val >= len(episodes):
                continue
            evidence.add(episodes[idx_val])
        
        final_episodes = set(evidence).union(retrieved_episodes)
        final_episodes = sorted(
            final_episodes,
            key=lambda e: (e.timestamp is None,
                        e.timestamp)
        )
        return {
            "is_sufficient": response.get("is_sufficient", False),
            "evidence": evidence,
            "new_query": response.get("new_query", query.query),
            "confidence_score": response.get("confidence_score", 0.0),
            "episodes": final_episodes,
            "input_token": input_token,
            "output_token": output_token,
        }

    async def _do_default_query(self, policy: QueryPolicy, query: QueryParam) -> tuple[list[Episode], dict[str, Any]]:
        q = copy.deepcopy(query)
        # TODO: make this self-adaptive
        # if q.limit >= 15:
        #     q.limit /= 3
        success = False
        max_retry = 60
        while not success:
            try:
                result, matrics = await super().do_query(policy, q)
                success = True
            except Exception as e:
                max_retry -= 1
                if max_retry == 0:
                    print(f"ERROR: Reranker failed after maximum retries.")
                    raise e
                if "ThrottlingException" in str(e):
                    print(f"WARNING: Reranker throttling exception, retrying after 5 second...")
                    time.sleep(5)
                else:
                    raise e
        return result, matrics

    async def do_query(
        self,
        policy: QueryPolicy,
        query: QueryParam,
    ) -> tuple[list[Episode], dict[str, Any]]:
        logger.info(f"CALLING {self.agent_name} with query: {query.query}")
        perf_matrics = self._init_perf_matrics()
        retrieved_evidence: set[Episode] = set()
        sufficiency_response: dict[str, Any] = {}
        used_query: list[str] = []

        curr_query = copy.deepcopy(query)
        for _ in range(self._max_attempts):
            curr_query.query = sufficiency_response.get("new_query", query.query)
            if curr_query.query in used_query or curr_query.query == "":
                # print("The model did not rewrite the query")
                break
            used_query.append(curr_query.query)
            # Step 1: Perform the query
            result, p_matrics = await self._do_default_query(policy, curr_query)
            self._update_perf_matrics(p_matrics, perf_matrics)

            # Step 2: Check if the evidence is enough to answer the original query
            llm_start = time.time()
            sufficiency_response = await self.combined_check_and_rewrite(query, result, retrieved_evidence, used_query)
            perf_matrics["llm_time"] += time.time() - llm_start
            retrieved_evidence.update(sufficiency_response["evidence"])

            perf_matrics["queries"].append(curr_query.query)
            perf_matrics["is_sufficient"].append(sufficiency_response["is_sufficient"])
            perf_matrics["evidence"].append([DeclarativeMemory.string_from_episode_context([e]) for e in sufficiency_response["evidence"]])
            perf_matrics["confidence_scores"].append(sufficiency_response["confidence_score"])
            perf_matrics["input_token"] += sufficiency_response["input_token"]
            perf_matrics["output_token"] += sufficiency_response["output_token"]
            if sufficiency_response["is_sufficient"] and sufficiency_response["confidence_score"] >= self._confidence_score:
                logger.debug("The default agent can answer the query with enough confidence")
                # print(f"Enough evidence with rewrites: {used_query}")
                break

        # Rerank base on all queries used
        q = copy.deepcopy(query)
        q.query = query.query + "\n".join(used_query)
        final_episodes = await self._do_rerank(q, sufficiency_response["episodes"])
        return final_episodes, perf_matrics
