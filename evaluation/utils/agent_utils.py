import os
import re
import time
from typing import Any

import boto3
import neo4j
import openai

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)
from memmachine.retrieval_agent.agents import (
    ChainOfQueryAgent,
    MemMachineAgent,
    SplitQueryAgent,
    ToolSelectAgent,
)
from memmachine.retrieval_agent.common.agent_api import (
    AgentToolBase,
    AgentToolBaseParam,
    QueryParam,
    QueryPolicy,
)


async def process_question(
    answer_prompt: str,
    query_agent: AgentToolBase,
    memory: DeclarativeMemory,
    model: openai.AsyncOpenAI,
    question: str,
    answer: str,
    category: int | str,
    supporting_facts: list[str],
    adversarial_answer: str = "",
    search_limit: int = 20,
    model_name: str = "gpt-5-mini",
    full_content: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
):
    perf_matrics: dict[str, Any] = {}
    memory_start = 0
    memory_end = 0
    prompt = ""
    formatted_context = ""
    chunks = []

    if full_content is None:
        memory_start = time.time()
        chunks, perf_matrics = await query_agent.do_query(
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

        formatted_context = memory.string_from_episode_context(chunks)
    else:
        formatted_context = full_content

    prompt = answer_prompt.format(memories=formatted_context, question=question)

    rsp_start = time.time()
    rsp = await model.responses.create(
        model=model_name,
        max_output_tokens=4096,
        top_p=1,
        input=[{"role": "user", "content": prompt}],
    )
    rsp_end = time.time()

    mem_retrieval_time = perf_matrics.get("memory_retrieval_time", 0)
    if mem_retrieval_time == 0:
        mem_retrieval_time = memory_end - memory_start
    llm_time = perf_matrics.get("llm_time", 0)
    print(
        f"Question: {question}\n"
        f"Agent used: {perf_matrics.get('agent', 'N/A')}\n"
        f"Memory search called: {perf_matrics.get('memory_search_called', 0)} times\n"
        f"Memory retrieval time: {mem_retrieval_time:.2f} seconds\n"
        f"LLM time for retrieval: {llm_time:.2f} seconds\n"
        f"LLM answering time: {rsp_end - rsp_start:.2f} seconds\n"
    )

    rsp_text = rsp.output_text
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

    res.update(perf_matrics)
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
            "tool_select_intput_token", 0
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


def init_vector_graph_store(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "neo4j_password",
) -> Neo4jVectorGraphStore:
    neo4j_driver = neo4j.AsyncGraphDatabase.driver(
        uri=neo4j_uri,
        auth=(
            neo4j_user,
            neo4j_password,
        ),
        # Default is 1 hour.
        max_connection_lifetime=7200,
        max_connection_pool_size=100,
        connection_acquisition_timeout=60.0,
        max_transaction_retry_time=15.0,
    )

    vector_graph_store = Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            max_concurrent_transactions=1000,
            range_index_hierarchies=[["uid"], ["timestamp", "uid"]],
            range_index_creation_threshold=100,
            vector_index_creation_threshold=100,
        )
    )
    return vector_graph_store


async def init_memmachine_params(
    vector_graph_store: Neo4jVectorGraphStore,
    model_name: str = "gpt-5-mini",
    session_id: str = "",
    agent_name: str = "ToolSelectAgent",
    message_sentence_chunking: bool = False,
) -> tuple[DeclarativeMemory, openai.AsyncOpenAI, AgentToolBase]:
    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embedder = OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model="text-embedding-3-small",
            dimensions=1536,
        )
    )

    region = "us-west-2"
    aws_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    reranker = AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=aws_client,
            region=region,
            model_id="amazon.rerank-v1:0",
        )
    )

    memory = DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id=session_id,
            vector_graph_store=vector_graph_store,
            embedder=embedder,
            reranker=reranker,
            message_sentence_chunking=message_sentence_chunking,
        )
    )

    agent_model: LanguageModel = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1",
            ),
            model=model_name,
            # Default medium for gpt-5-mini
            # reasoning_effort="minimal",
        ),
    )
    query_agent = await init_agent(agent_model, reranker, agent_name)

    answer_model = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return memory, answer_model, query_agent
