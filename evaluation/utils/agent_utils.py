import re
import time
from pathlib import Path
from typing import Any

from memmachine_server.common.configuration import Configuration
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.metrics_factory import PrometheusMetricsFactory
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.common.resource_manager.resource_manager import (
    ResourceManagerImpl,
)
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.episodic_memory.episodic_memory import (
    EpisodicMemoryParams,
)
from memmachine_server.episodic_memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryParams,
)
from memmachine_server.retrieval_agent.agents import (
    ChainOfQueryAgent,
    MemMachineAgent,
    SplitQueryAgent,
    ToolSelectAgent,
)
from memmachine_server.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)


def load_eval_config(config_path: str) -> ResourceManagerImpl:
    """Load configuration.yml and return an initialized ResourceManagerImpl.

    Args:
        config_path: Path to the configuration YAML file.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"configuration.yml not found at '{config_path}'.\n"
            "Please place a configuration.yml in the retrieval_agent directory "
            "before running benchmarks. See evaluation/retrieval_agent/README.md "
            "for details."
        )
    config = Configuration.load_yml_file(str(config_file))
    return ResourceManagerImpl(config)


async def process_question(
    answer_prompt: str,
    query_agent: AgentToolBase,
    memory: EpisodicMemory,
    answer_model: LanguageModel,
    question: str,
    answer: str,
    category: int | str,
    supporting_facts: list[str],
    adversarial_answer: str = "",
    search_limit: int = 20,
    full_content: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
):
    perf_metrics: dict[str, Any] = {}
    memory_start = 0
    memory_end = 0
    formatted_context = ""
    chunks = []

    if full_content is None:
        memory_start = time.time()
        chunks, perf_metrics = await query_agent.do_query(
            QueryPolicy(
                token_cost=10,
                time_cost=10,
                accuracy_score=10,
                confidence_score=10,
                max_attempts=3,
                max_return_len=10000,
            ),
            QueryParam(query=question, limit=search_limit, memory=memory),
        )
        memory_end = time.time()

        formatted_context = episodes_to_string(chunks)
    else:
        formatted_context = full_content

    prompt = answer_prompt.format(memories=formatted_context, question=question)

    rsp_start = time.time()
    rsp_text, _ = await answer_model.generate_response(user_prompt=prompt)
    rsp_end = time.time()

    mem_retrieval_time = perf_metrics.get("memory_retrieval_time", 0)
    if mem_retrieval_time == 0:
        mem_retrieval_time = memory_end - memory_start
    llm_time = perf_metrics.get("llm_time", 0)
    print(
        f"Question: {question}\n"
        f"Agent used: {perf_metrics.get('agent', 'N/A')}\n"
        f"Memory search called: {perf_metrics.get('memory_search_called', 0)} times\n"
        f"Memory retrieval time: {mem_retrieval_time:.2f} seconds\n"
        f"LLM time for retrieval: {llm_time:.2f} seconds\n"
        f"LLM answering time: {rsp_end - rsp_start:.2f} seconds\n"
    )

    res = {
        "question": question,
        "golden_answer": answer,
        "model_answer": rsp_text,
        "category": category,
        "supporting_facts": supporting_facts,
        "adversarial_answer": adversarial_answer,
        "conversation_memories": formatted_context,
        "num_episodes_retrieved": len(chunks),
    }

    res.update(perf_metrics)
    res.update(extra_attributes or {})

    return category, res


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _match_tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 3}


def _fact_variants(fact: str) -> list[str]:
    variants = [fact.strip()]
    if ":" in fact:
        sent_part = fact.split(":", 1)[1].strip()
        if sent_part:
            variants.append(sent_part)
    return [v for v in variants if v]


def _fact_in_mem(fact: str, mem: str, mem_lines_norm: list[str]) -> bool:
    mem_norm = _normalize_for_match(mem)
    for variant in _fact_variants(fact):
        variant_norm = _normalize_for_match(variant)
        if variant_norm and variant_norm in mem_norm:
            return True

        # OpenClaw search snippets may be shortened; allow conservative overlap.
        variant_tokens = _match_tokens(variant_norm)
        if len(variant_tokens) < 5:
            continue
        for line in mem_lines_norm:
            line_tokens = _match_tokens(line)
            if len(line_tokens) < 5:
                continue
            overlap = len(variant_tokens & line_tokens)
            overlap_ratio = overlap / len(variant_tokens)
            if overlap_ratio >= 0.6:
                return True

    return False


def init_attribute_matrix() -> dict[str, Any]:
    return {
        "customize_attributes": {},  # dict[str, Any] for different dataset use
        "tools_called": {},  # dict[str, int]
        "tools_hits": {},  # dict[str, int]
        "tools_facts": {},  # dict[str, int]
        "tools_episodes": {},  # dict[str, int]
        "tools_input_tokens": {
            "ToolSelectAgent": 0,
        },  # dict[str, int]
        "tools_output_tokens": {
            "ToolSelectAgent": 0,
        },  # dict[str, int]
        "num_facts": 0,
        "num_hits": 0,
        "num_episodes_retrieved": 0,
        "num_questions": 0,
        "memory_retrieval_time_total": 0.0,
        "llm_time_total": 0.0,
        "question_used_llm_total": 0,
    }


def update_results(
    responses: list[tuple[str, dict[str, Any]]],
    attribute_matrix: dict[str, Any],
    results: dict[str, Any],
):
    for category, response in responses:
        attribute_matrix["num_questions"] += 1
        tool = response.get("selected_tool", "Unknown")
        if tool not in attribute_matrix["tools_hits"]:
            attribute_matrix["tools_hits"][tool] = 0
            attribute_matrix["tools_facts"][tool] = 0
            attribute_matrix["tools_episodes"][tool] = 0
            attribute_matrix["tools_called"][tool] = 0
            attribute_matrix["tools_input_tokens"][tool] = 0
            attribute_matrix["tools_output_tokens"][tool] = 0

        mem = response["conversation_memories"]
        mem_lines_norm = (
            [_normalize_for_match(line) for line in mem.splitlines() if line]
            if isinstance(mem, str)
            else []
        )
        fact_hits = []
        fact_miss = []
        for fact in response["supporting_facts"]:
            if (
                isinstance(mem, str)
                and isinstance(fact, str)
                and _fact_in_mem(fact, mem, mem_lines_norm)
            ):
                attribute_matrix["tools_hits"][tool] += 1
                fact_hits.append(f"[HIT] {fact}\n")
            else:
                fact_miss.append(f"[MISS] {fact}\n")

        response["fact_hits"] = fact_hits
        response["fact_miss"] = fact_miss

        attribute_matrix["num_hits"] += len(response["fact_hits"])
        attribute_matrix["num_facts"] += len(response["supporting_facts"])
        attribute_matrix["tools_facts"][tool] += len(response["supporting_facts"])
        attribute_matrix["num_episodes_retrieved"] += response["num_episodes_retrieved"]
        attribute_matrix["tools_episodes"][tool] += response["num_episodes_retrieved"]
        attribute_matrix["tools_called"][tool] += 1
        attribute_matrix["tools_input_tokens"][tool] += response.get("input_token", 0)
        attribute_matrix["tools_output_tokens"][tool] += response.get("output_token", 0)
        attribute_matrix["tools_input_tokens"]["ToolSelectAgent"] += response.get(
            "tool_select_input_token", 0
        )
        attribute_matrix["tools_output_tokens"]["ToolSelectAgent"] += response.get(
            "tool_select_output_token", 0
        )
        attribute_matrix["memory_retrieval_time_total"] += response.get(
            "memory_retrieval_time", 0
        )
        attribute_matrix["llm_time_total"] += response.get("llm_time", 0)
        if response.get("llm_time", 0) > 0:
            attribute_matrix["question_used_llm_total"] += 1

        category_result = results.get(category, [])
        category_result.append(response)
        results[category] = category_result


def update_final_attribute_matrix(
    test_preffix: str,
    attribute_matrix: dict[str, Any],
    results: dict[str, Any],
):
    num_hits = attribute_matrix["num_hits"]
    num_facts = attribute_matrix["num_facts"]
    num_episodes_retrieved = attribute_matrix["num_episodes_retrieved"]
    tools_called = attribute_matrix["tools_called"]
    tools_hits = attribute_matrix["tools_hits"]
    tools_facts = attribute_matrix["tools_facts"]
    tools_episodes = attribute_matrix["tools_episodes"]
    tools_input_tokens = attribute_matrix["tools_input_tokens"]
    tools_output_tokens = attribute_matrix["tools_output_tokens"]
    num_questions = attribute_matrix["num_questions"]
    memory_retrieval_time_avg = (
        attribute_matrix["memory_retrieval_time_total"] / num_questions
        if num_questions > 0
        else 0.0
    )
    llm_time_avg = (
        attribute_matrix["llm_time_total"] / attribute_matrix["question_used_llm_total"]
        if attribute_matrix["question_used_llm_total"] > 0
        else 0.0
    )

    recall = (
        f"{num_hits}/{num_facts} = {num_hits / num_facts * 100:.2f}%"
        if num_facts > 0
        else "N/A"
    )
    precision = (
        f"{num_hits}/{num_episodes_retrieved} = {num_hits / num_episodes_retrieved * 100:.2f}%"
        if num_episodes_retrieved > 0
        else "N/A"
    )
    average_episodes_retrieved = (
        num_episodes_retrieved / num_questions if num_questions > 0 else 0.0
    )
    tools_report = ""
    for tool in tools_called:
        tool_recall = (
            f"{tools_hits[tool]}/{tools_facts[tool]} = {tools_hits[tool] / tools_facts[tool] * 100:.2f}%"
            if tools_facts[tool] > 0
            else "N/A"
        )
        tool_precision = (
            f"{tools_hits[tool]}/{tools_episodes[tool]} = {tools_hits[tool] / tools_episodes[tool] * 100:.2f}%"
            if tools_episodes[tool] > 0
            else "N/A"
        )
        tools_report += f"""Tool: {tool}
    Recall: {tool_recall}
    Precision: {tool_precision}
    Avg Episodes Retrieved per Question: {tools_episodes[tool] / tools_called[tool]:.2f}
    Avg Input Tokens per Question: {tools_input_tokens[tool] / tools_called[tool]:.2f}
    Avg Output Tokens per Question: {tools_output_tokens[tool] / tools_called[tool]:.2f}
"""

    customize_msgs = None
    customize_attributes = attribute_matrix["customize_attributes"]
    for key, val in customize_attributes.items():
        if customize_msgs is None:
            customize_msgs = ""
        if isinstance(val, float):
            val = round(val, 3)
        customize_msgs += f"{key}: {val}\n"

    final_matrix = f"""{test_preffix} Recall: {recall}
{test_preffix} Precision: {precision}
{test_preffix} Average Episodes Retrieved per Question: {average_episodes_retrieved:.2f}
{test_preffix} Average Memory Retrieval Time per Question: {memory_retrieval_time_avg:.2f} seconds
{test_preffix} Average LLM Time per Question (only for questions that used LLM): {llm_time_avg:.2f} seconds
{tools_report}
ToolSelectAgent Avg Input Tokens per Question: {tools_input_tokens["ToolSelectAgent"] / num_questions:.2f}
ToolSelectAgent Avg Output Tokens per Question: {tools_output_tokens["ToolSelectAgent"] / num_questions:.2f}
{customize_msgs if customize_msgs is not None else ""}
"""

    matrix_name = f"{test_preffix}_final_matrix"
    for res_list in results.values():
        res_list[0][matrix_name] = final_matrix
        break
    return final_matrix


async def init_agent(
    model: LanguageModel,
    reranker: Reranker,
    agent_name: str,
) -> AgentToolBase:
    param: AgentToolBaseParam = AgentToolBaseParam(
        model=None,
        children_tools=[],
        extra_params={},
        reranker=reranker,
    )
    memory_agent: MemMachineAgent = MemMachineAgent(param)
    if agent_name == memory_agent.agent_name:
        return memory_agent

    param: AgentToolBaseParam = AgentToolBaseParam(
        model=model, children_tools=[memory_agent], extra_params={}, reranker=reranker
    )

    coq_agent: ChainOfQueryAgent = ChainOfQueryAgent(param)
    split_agent: SplitQueryAgent = SplitQueryAgent(param)

    if agent_name == coq_agent.agent_name:
        return coq_agent
    if agent_name == split_agent.agent_name:
        return split_agent

    param: AgentToolBaseParam = AgentToolBaseParam(
        model=model,
        children_tools=[split_agent, coq_agent, memory_agent],
        extra_params={"default_tool_name": coq_agent.agent_name},
    )

    select_agent: ToolSelectAgent = ToolSelectAgent(param)

    return select_agent


async def init_memmachine_params(
    resource_manager: ResourceManagerImpl,
    session_id: str = "",
    agent_name: str = "ToolSelectAgent",
    message_sentence_chunking: bool = False,
) -> tuple[EpisodicMemory, LanguageModel, AgentToolBase, str, str]:
    """Initialize MemMachine components from a ResourceManagerImpl.

    Components are resolved from the loaded configuration:

    - Embedder:           ``episodic_memory.long_term_memory.embedder``
    - Reranker:           ``retrieval_agent.reranker`` (fallback:
                          ``episodic_memory.long_term_memory.reranker``)
    - Vector graph store: ``episodic_memory.long_term_memory.vector_graph_store``
    - Agent model:        ``retrieval_agent.llm_model``
    - Answer model:       ``retrieval_agent.answer_llm_model`` (falls back to ``llm_model``)

    Returns:
        A tuple of (memory, answer_model, query_agent, agent_model_id, answer_model_id).

        The model ID strings (``agent_model_id``, ``answer_model_id``) are returned
        for logging purposes - they are stored in evaluation results to track which
        models were used for each experiment.
    """
    conf = resource_manager.config
    ltm_conf = conf.episodic_memory.long_term_memory
    if ltm_conf is None:
        raise ValueError(
            "episodic_memory.long_term_memory is not configured in configuration.yml"
        )

    embedder_id = ltm_conf.embedder
    if not embedder_id:
        raise ValueError(
            "episodic_memory.long_term_memory.embedder is not set in configuration.yml"
        )
    embedder = await resource_manager.get_embedder(embedder_id)

    reranker_id = conf.retrieval_agent.reranker or ltm_conf.reranker
    if not reranker_id:
        raise ValueError(
            "Neither retrieval_agent.reranker nor "
            "episodic_memory.long_term_memory.reranker is set in configuration.yml"
        )
    reranker = await resource_manager.get_reranker(reranker_id)

    vector_graph_store_id = ltm_conf.vector_graph_store
    if not vector_graph_store_id:
        raise ValueError(
            "episodic_memory.long_term_memory.vector_graph_store is not set in "
            "configuration.yml"
        )
    vector_graph_store = await resource_manager.get_vector_graph_store(
        vector_graph_store_id
    )

    agent_model_id = conf.retrieval_agent.llm_model
    if not agent_model_id:
        raise ValueError("retrieval_agent.llm_model is not set in configuration.yml")
    agent_model = await resource_manager.get_language_model(agent_model_id)

    # Answer model: use answer_llm_model if set, otherwise fall back to llm_model
    answer_model_id = (
        conf.retrieval_agent.answer_llm_model or conf.retrieval_agent.llm_model
    )
    answer_model = await resource_manager.get_language_model(answer_model_id)

    normalized_session_id = session_id or "evaluation_session"

    long_term_memory = LongTermMemory(
        LongTermMemoryParams(
            session_id=normalized_session_id,
            vector_graph_store=vector_graph_store,
            embedder=embedder,
            reranker=reranker,
            message_sentence_chunking=message_sentence_chunking,
        )
    )
    memory = EpisodicMemory(
        EpisodicMemoryParams(
            session_key=normalized_session_id,
            metrics_factory=PrometheusMetricsFactory(),
            long_term_memory=long_term_memory,
            short_term_memory=None,
            enabled=True,
        ),
    )

    query_agent = await init_agent(agent_model, reranker, agent_name)

    return memory, answer_model, query_agent, agent_model_id, answer_model_id
