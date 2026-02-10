import asyncio
import logging
from typing import Any, cast

from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryPolicy,
    QueryParam,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.episodic_memory.declarative_memory import Episode

logger = logging.getLogger(__name__)

# Citation: Luo et al. (2025), "Agent Lightning: Train ANY AI Agents with
# Reinforcement Learning", arXiv:2508.03680.
TOOL_SELECT_PROMPT = """You are a tool router. Your task is to select exactly ONE tool name from the provided list that best fits the user query. Do not call any tools. Use only the text in {query}; d
o not assume external context or missing metadata. You may interpret dependency only from explicit linguistic structure (e.g., “use X to find Y”, possessive/relationship chains)
.

GOAL
- Choose exactly one of: {coq}, {split_query}, {memory_retrieval}
- Output NONE only when the query type cannot be determined from {query} (e.g., empty/invalid/malformed).

MECHANISM (what to do, then how to do it)

1) Validate input
- If {query} is empty, whitespace-only, null-like, non-linguistic garbage, or otherwise not classifiable as a query/request → output NONE.

2) Classify the query type from {query} using ONLY the operational criteria below (pick exactly one):

A) MULTI-HOP (dependency chain; later step depends on earlier result)
Choose MULTI-HOP if the query requires two or more dependent steps where you must first determine X, then use X to determine Y (or more).
- Explicit dependency signals: “then”, “after”, “using that”, “based on that result”, “from there”, “which of those”, “once you find”, “given the answer to”, “trace/derive”.
- Relationship-chain dependency: queries that require resolving an intermediate entity/attribute to proceed (e.g., “X’s spouse’s company”, “author of the book that inspired the
film”).
- Comparison/timeline becomes MULTI-HOP ONLY if it requires derived attributes that must be found first (e.g., “Which is older, the CEO of A or the founder of B?” requires findi
ng CEO/founder first).
- Tie-breaker: If any explicit dependency chain exists, classify as MULTI-HOP even if multiple entities/keywords appear.

B) SINGLE-HOP WITH MULTIPLE ENTITIES/KEYWORDS (independent sub-queries; no dependency)
Choose this when the query contains multiple entities/subjects that can be answered via separate independent lookups and then combined, without needing any earlier result to for
m the later lookup.
- Signals: “A and B”, “A, B, and C”, “for each of”, “separately”, “list … for these items”.
- Comparisons are HERE when both sides are directly look-up-able without intermediate derivations (e.g., “Compare the populations of France and Germany”).
- Multiple distinct questions in one message are HERE if they are independent (e.g., “What is X? Also, what is Y?”).

C) SINGLE-HOP / DIRECT (one main subject; no splitting needed)
Choose this when the query is a single straightforward request about one primary subject/lookup, not requiring decomposition or splitting.

3) Deterministic tool selection (hard mapping)
- If MULTI-HOP → output {coq}
- Else if SINGLE-HOP WITH MULTIPLE ENTITIES/KEYWORDS → output {split_query}
- Else if SINGLE-HOP / DIRECT → output {memory_retrieval}
- Else → output NONE

AVAILABLE TOOLS (names must match exactly; do not assume extra metadata)
{coq}: Chain of Query agent that can decompose complex multi-hop queries into multiple simple single-hop queries and search them step by step.
{split_query}: Split Query agent that can split a single-hop query with multiple entities/keywords into multiple single-hop queries and search them separately.
{memory_retrieval}: Memory Retrieval agent that searches the query without modifying it.

INPUTS
Query:
{query}

OUTPUT FORMAT (strict)
- Output a single line containing only one of:
  - a tool name exactly matching: {coq} OR {split_query} OR {memory_retrieval}
  - NONE

CLASSIFICATION EXAMPLES (for calibration; do not output these)
1) “Who is the author of ‘Dune’?” → {memory_retrieval}
2) “Give the capitals of Spain and Portugal.” → {split_query}
3) “Find the spouse of Marie Curie, then name his primary field.” → {coq}
4) “Compare GDP per capita of Japan and South Korea.” → {split_query}
5) “Which is older: the founder of Company A or the CEO of Company B?” → {coq}
"""

class ToolSelectAgent(AgentToolBase):
    def __init__(self, param: AgentToolBaseParam):
        super().__init__(param)
        if self._model is None:
            raise ValueError("Model is not set")
        self._extra_param = param.extra_params or {}
        self._tool_select_prompt = self._extra_param.get("tool_select_prompt", TOOL_SELECT_PROMPT)
        default_tool_name = self._extra_param.get("default_tool_name")
        self._default_tool: AgentToolBase | None = None
        # FIXME: Consider use global variables or import tool names. We need to do this because
        # tool descriptions cannot be tuned independently without the entire prompt.
        self._coq_agent: AgentToolBase | None = None
        self._split_agent: AgentToolBase | None = None
        self._memory_agent: AgentToolBase | None = None
        for tool in self._children_tools:
            if tool.agent_name == "ChainOfQueryAgent":
                self._coq_agent = tool
            elif tool.agent_name == "SplitQueryAgent":
                self._split_agent = tool
            elif tool.agent_name == "MemMachineAgent":
                self._memory_agent = tool
            if tool.agent_name == default_tool_name:
                self._default_tool = tool
        if self._coq_agent is None or self._split_agent is None or self._memory_agent is None:
            raise ValueError("Tool select agent requires 'ChainOfQueryAgent', 'SplitQueryAgent' and 'MemMachineAgent' tools as child tools")

    @property
    def agent_name(self) -> str:
        return "ToolSelectAgent"

    @property
    def agent_description(self) -> str:
        return """This agent selects tools from a list of avaible tools to perform the query. The selection
        is based on a few factors, including: the query, the accuracy requirement, the token cost constraint,
        the time cost constraint and the functionalities of available tools. It aggregates the result from each tool and return the final result.
        """

    @property
    def accuracy_score(self) -> int:
        return 8

    @property
    def token_cost(self) -> int:
        return 6

    @property
    def time_cost(self) -> int:
        return 8

    def add_tool(self, tool: AgentToolBase):
        super().add_tool(tool)

    async def _select_tool_by_model(self, policy: QueryPolicy, query: QueryParam ) -> list[AgentToolBase]:
        prompt = self._tool_select_prompt.format(
            query=query,
            coq=self._coq_agent.agent_name,
            split_query=self._split_agent.agent_name,
            memory_retrieval=self._memory_agent.agent_name,
        )
        model = cast(LanguageModel, self._model)
        rsp, _, input_token, output_token = await model.generate_response_with_token_usage(user_prompt=prompt)
        logger.debug(f"Selected tools: {rsp}")

        selected_tool = None
        for tool in self._children_tools:
            if tool.agent_name in rsp:
                selected_tool = tool
                break
        if selected_tool is None:
            logger.warning(f"Tool {rsp} not found")
        if selected_tool is None and self._default_tool is not None:
            logger.warning("No tool selected, using default tool")
            return [self._default_tool]
        return selected_tool, input_token, output_token

    async def do_query(self, policy: QueryPolicy, query: QueryParam) -> tuple[list[Episode], dict[str, Any]]:
        logger.info(f"CALLING {self.agent_name} with query: {query.query}")
        tool, input_token, output_token = await self._select_tool_by_model(policy, query)
        if tool is None:
            if self._default_tool is not None:
                tool = self._default_tool
            else:
                raise RuntimeError("No tool selected")
        chunks, perf_matrics = await tool.do_query(policy, query)
        perf_matrics["selected_tool"] = tool.agent_name
        perf_matrics["tool_select_intput_token"] = input_token
        perf_matrics["tool_select_output_token"] = output_token
        return chunks, perf_matrics
