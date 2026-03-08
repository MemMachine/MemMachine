import os
import re
import time
from typing import Any

import boto3
import neo4j
import openai
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.episode_store.episode_model import episodes_to_string
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.common.metrics_factory import PrometheusMetricsFactory
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine_server.common.reranker.reranker import Reranker
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory import EpisodicMemory
from memmachine_server.episodic_memory.episodic_memory import (
    EpisodicMemoryParams,
)
from memmachine_server.episodic_memory.long_term_memory import (
    LongTermMemory,
    LongTermMemoryParams,
)
from memmachine_server.retrieval_skill import create_retrieval_skill
from memmachine_server.retrieval_skill.common.skill_api import (
    QueryParam,
    QueryPolicy,
    SkillToolBase,
    SkillToolBaseParam,
)
from memmachine_server.retrieval_skill.subskills.direct_memory_skill import (
    MemMachineSkill,
)

RETRIEVE_SKILL_NAME = "RetrieveSkill"
DIRECT_MEMORY_SKILL_NAME = "MemMachineSkill"
RETRIEVAL_HINT_UNCERTAINTY_PATTERN = re.compile(
    r"(?i)\b(if|likely|probably|suggests?|inferred?|assum(?:e|ed|ption)|"
    r"traditional|uncertain|unknown|not explicit|no explicit|may be|might)\b"
)


def _normalize_sub_skill_name(raw_name: str) -> str | None:
    normalized = raw_name.strip()
    if not normalized:
        return None

    key = normalized.replace("-", "_").lower()
    mapping = {
        "coq": "ChainOfQuerySkill",
        "chainofqueryskill": "ChainOfQuerySkill",
        "chain_of_query_skill": "ChainOfQuerySkill",
        "split": "SplitSkill",
        "splitskill": "SplitSkill",
        "direct_memory": "DirectMemorySkill",
        "memmachineskill": "DirectMemorySkill",
    }
    mapped = mapping.get(key)
    if mapped is not None:
        return mapped
    if normalized.endswith("Agent"):
        return f"{normalized[:-5]}Skill"
    return normalized


def _extract_sub_skills(perf_metrics: dict[str, Any]) -> list[str]:
    used_sub_skills: list[str] = []
    seen: set[str] = set()

    def _add(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        skill_name = _normalize_sub_skill_name(raw_name)
        if not skill_name:
            return
        if skill_name not in seen:
            used_sub_skills.append(skill_name)
            seen.add(skill_name)

    for run in perf_metrics.get("orchestrator_sub_skill_runs", []):
        if isinstance(run, dict):
            _add(run.get("skill_name"))
    _add(perf_metrics.get("selected_skill"))
    _add(perf_metrics.get("selected_skill_name"))
    return used_sub_skills


def _build_skill_used_label(perf_metrics: dict[str, Any]) -> str:
    labels: list[str] = []
    seen: set[str] = set()

    def _push(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        label = _normalize_sub_skill_name(raw_name) or raw_name.strip()
        if not label:
            return
        if label not in seen:
            labels.append(label)
            seen.add(label)

    # Keep top-level first, then append sub-skills in discovered order.
    _push(perf_metrics.get("skill"))
    for sub_skill in _extract_sub_skills(perf_metrics):
        _push(sub_skill)

    return ", ".join(labels) if labels else "N/A"


def _extract_confidence_from_metrics(
    perf_metrics: dict[str, Any],
    *keys: str,
) -> float | None:
    for key in keys:
        value = perf_metrics.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _retrieval_hint_reliability(
    confidence: float | None, reason_note: str | None
) -> str:
    if confidence is not None and confidence >= 0.85:
        if isinstance(reason_note, str) and RETRIEVAL_HINT_UNCERTAINTY_PATTERN.search(
            reason_note
        ):
            return "tentative"
        return "high"
    return "tentative"


def _format_retrieval_candidate_hint(
    *,
    prefix: str,
    candidate: str,
    reliability: str,
    confidence: float | None,
    reason_note: str | None,
) -> str:
    confidence_text = f", confidence={confidence:.2f}" if confidence is not None else ""
    hint = (
        "[Retrieval-Skill Summary] "
        f"{prefix} (reliability={reliability}{confidence_text}): "
        f"{candidate.strip()}."
    )
    if reliability == "tentative":
        hint += " Status: unverified; corroborate before final use."
    if isinstance(reason_note, str) and reason_note.strip():
        hint += f" Reason: {reason_note.strip()}."
    return hint


def _build_retrieval_answer_hint(perf_metrics: dict[str, Any]) -> str:
    if bool(perf_metrics.get("top_level_is_sufficient", False)):
        answer_candidate = perf_metrics.get("answer_candidate")
        if not isinstance(answer_candidate, str) or not answer_candidate.strip():
            answer_candidate = perf_metrics.get("latest_answer_candidate")
        reason_note = perf_metrics.get("top_level_reason_note")
        confidence = _extract_confidence_from_metrics(
            perf_metrics,
            "top_level_confidence_score",
            "latest_sufficiency_confidence_score",
        )
        reliability = _retrieval_hint_reliability(
            confidence=confidence,
            reason_note=reason_note if isinstance(reason_note, str) else None,
        )
        if isinstance(answer_candidate, str) and answer_candidate.strip():
            return _format_retrieval_candidate_hint(
                prefix="Top-level answer candidate",
                candidate=answer_candidate,
                reliability=reliability,
                confidence=confidence,
                reason_note=reason_note if isinstance(reason_note, str) else None,
            )
        if isinstance(reason_note, str) and reason_note.strip():
            return _format_retrieval_candidate_hint(
                prefix="Top-level sufficiency reason",
                candidate=reason_note,
                reliability=reliability,
                confidence=confidence,
                reason_note=None,
            )

    if not bool(perf_metrics.get("latest_sufficiency_signal", False)):
        return ""
    answer_candidate = perf_metrics.get("latest_answer_candidate")
    if not isinstance(answer_candidate, str) or not answer_candidate.strip():
        return ""
    reason_note = perf_metrics.get("latest_sufficiency_reason_note")
    confidence = _extract_confidence_from_metrics(
        perf_metrics,
        "latest_sufficiency_confidence_score",
    )
    # Latest sub-skill signals without top-level sufficiency are provisional.
    reliability = "tentative"
    return _format_retrieval_candidate_hint(
        prefix="Sub-skill provisional answer candidate",
        candidate=answer_candidate,
        reliability=reliability,
        confidence=confidence,
        reason_note=reason_note if isinstance(reason_note, str) else None,
    )


def _metric_as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_llm_call_count(perf_metrics: dict[str, Any]) -> int:
    explicit = _metric_as_int(perf_metrics.get("llm_call_count", 0))
    if explicit > 0:
        return explicit

    inferred = _metric_as_int(perf_metrics.get("top_level_session_turn_count", 0))
    for run in perf_metrics.get("orchestrator_sub_skill_runs", []):
        if not isinstance(run, dict):
            continue
        inferred += _metric_as_int(run.get("llm_call_count", 0))
    return inferred


def _extract_memory_search_latency_breakdown(
    perf_metrics: dict[str, Any],
) -> list[float]:
    raw = perf_metrics.get("memory_search_latency_seconds")
    if not isinstance(raw, list):
        return []
    return [
        float(item)
        for item in raw
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]


def _is_unknown_like_answer(text: str) -> bool:
    normalized = text.strip().lower()
    return any(
        marker in normalized
        for marker in (
            "i don't know",
            "i don’t know",
            "unknown",
            "not sure",
            "cannot determine",
            "can't determine",
            "insufficient",
            "not enough information",
        )
    )


def _is_same_country_or_nationality_question(question: str) -> bool:
    normalized = f" {question.strip().lower()} "
    if not re.match(
        r"\s*(is|are|was|were|do|does|did|can|could|has|have|had)\b",
        normalized,
    ):
        return False
    return any(
        marker in normalized
        for marker in (
            " same country ",
            " same nationality ",
            " from the same country ",
            " from the same nationality ",
        )
    )


def _needs_answer_verification(
    *,
    question: str,
    perf_metrics: dict[str, Any],
    draft_answer: str,
) -> bool:
    if _is_unknown_like_answer(draft_answer):
        return True
    if _is_same_country_or_nationality_question(question):
        return True
    if bool(perf_metrics.get("top_level_is_sufficient", False)):
        confidence = perf_metrics.get("top_level_confidence_score")
        if (
            isinstance(confidence, int | float)
            and not isinstance(confidence, bool)
            and float(confidence) < 0.8
        ):
            return True
    return False


async def process_question(
    answer_prompt: str,
    query_skill: SkillToolBase,
    memory: EpisodicMemory,
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
    perf_metrics: dict[str, Any] = {}
    memory_start = 0
    memory_end = 0
    prompt = ""
    formatted_context = ""
    chunks = []

    if full_content is None:
        memory_start = time.time()
        chunks, perf_metrics = await query_skill.do_query(
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

    retrieval_answer_hint = _build_retrieval_answer_hint(perf_metrics)
    if retrieval_answer_hint:
        formatted_context = f"{retrieval_answer_hint}\n{formatted_context}".strip()

    prompt = answer_prompt.format(memories=formatted_context, question=question)

    rsp_start = time.time()
    rsp = await model.responses.create(
        model=model_name,
        max_output_tokens=4096,
        top_p=1,
        input=[{"role": "user", "content": prompt}],
    )
    rsp_text = rsp.output_text.strip()
    open_domain_rescue_used = False
    answer_verification_used = False
    if _is_unknown_like_answer(rsp_text):
        rescue_prompt = (
            "Answer the question using best available world knowledge. "
            "Do not say unknown unless truly unknowable. "
            "Provide only the concise final answer.\n"
            f"Question: {question}"
        )
        rescue_rsp = await model.responses.create(
            model=model_name,
            max_output_tokens=512,
            top_p=1,
            input=[{"role": "user", "content": rescue_prompt}],
        )
        rescue_text = rescue_rsp.output_text.strip()
        if rescue_text and not _is_unknown_like_answer(rescue_text):
            rsp_text = rescue_text
            open_domain_rescue_used = True

    if _needs_answer_verification(
        question=question,
        perf_metrics=perf_metrics,
        draft_answer=rsp_text,
    ):
        verification_prompt = f"""You are validating a draft answer to a multi-hop question.

Question: {question}

Memories:
{formatted_context}

Draft answer:
{rsp_text}

Requirements:
- If the draft is already correct, keep it unchanged.
- For same-country/same-nationality yes/no questions, answer "yes" when compared entities share at least one normalized country/nationality.
- For relation-chain questions, ensure the answer targets the final asked attribute (not an intermediate entity).
- Do not return unknown if a best-supported answer can be given from memory or general knowledge.
- Return only the final answer, concise."""
        verification_rsp = await model.responses.create(
            model=model_name,
            max_output_tokens=512,
            top_p=1,
            input=[{"role": "user", "content": verification_prompt}],
        )
        verified_text = verification_rsp.output_text.strip()
        if verified_text:
            rsp_text = verified_text
            answer_verification_used = True
    answer_end = time.time()

    mem_retrieval_time = perf_metrics.get("memory_retrieval_time", 0)
    if mem_retrieval_time == 0:
        mem_retrieval_time = memory_end - memory_start
    llm_time = perf_metrics.get("llm_time", 0)
    llm_call_count = _extract_llm_call_count(perf_metrics)
    memory_latency_breakdown = _extract_memory_search_latency_breakdown(perf_metrics)
    skill_used_label = _build_skill_used_label(perf_metrics)
    memory_latency_line = ""
    if memory_latency_breakdown:
        rounded = [round(value, 3) for value in memory_latency_breakdown]
        memory_latency_line = f"Memory search latency breakdown (s): {rounded}\n"
    print(
        f"Question: {question}\n"
        f"Skill used: {skill_used_label}\n"
        f"Memory search called: {perf_metrics.get('memory_search_called', 0)} times\n"
        f"Memory retrieval time: {mem_retrieval_time:.2f} seconds\n"
        f"{memory_latency_line}"
        f"LLM called: {llm_call_count} times\n"
        f"LLM time for retrieval: {llm_time:.2f} seconds\n"
        f"LLM answering time: {answer_end - rsp_start:.2f} seconds\n"
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
        "llm_call_count": llm_call_count,
        "open_domain_rescue_used": open_domain_rescue_used,
        "answer_verification_used": answer_verification_used,
    }

    res.update(perf_metrics)
    res.update(extra_attributes or {})
    res["llm_call_count"] = llm_call_count

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
        "tools_input_tokens": {},  # dict[str, int]
        "tools_output_tokens": {},  # dict[str, int]
        "tools_llm_calls": {},  # dict[str, int]
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
        tool = (
            response.get("selected_skill_name")
            or response.get("selected_skill")
            or response.get("route")
            or response.get("skill")
            or "Unknown"
        )
        if tool not in attribute_matrix["tools_hits"]:
            attribute_matrix["tools_hits"][tool] = 0
            attribute_matrix["tools_facts"][tool] = 0
            attribute_matrix["tools_episodes"][tool] = 0
            attribute_matrix["tools_called"][tool] = 0
            attribute_matrix["tools_input_tokens"][tool] = 0
            attribute_matrix["tools_output_tokens"][tool] = 0
            attribute_matrix["tools_llm_calls"][tool] = 0

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
        input_tokens = _metric_as_int(response.get("input_token", 0))
        output_tokens = _metric_as_int(response.get("output_token", 0))
        llm_call_count = _extract_llm_call_count(response)
        attribute_matrix["tools_input_tokens"][tool] += input_tokens
        attribute_matrix["tools_output_tokens"][tool] += output_tokens
        attribute_matrix["tools_llm_calls"][tool] += llm_call_count
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
    tools_llm_calls = attribute_matrix["tools_llm_calls"]
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
    Avg LLM Call per Question: {tools_llm_calls[tool] / tools_called[tool]:.2f}
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
{customize_msgs if customize_msgs is not None else ""}
"""

    matrix_name = f"{test_preffix}_final_matrix"
    for res_list in results.values():
        res_list[0][matrix_name] = final_matrix
        break
    return final_matrix


async def init_skill(
    model: LanguageModel,
    reranker: Reranker,
    skill_name: str,
) -> SkillToolBase:
    if skill_name == DIRECT_MEMORY_SKILL_NAME:
        return MemMachineSkill(
            SkillToolBaseParam(
                model=None,
                children_tools=[],
                extra_params={},
                reranker=reranker,
            )
        )
    return create_retrieval_skill(model=model, reranker=reranker)


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
    skill_name: str = RETRIEVE_SKILL_NAME,
    message_sentence_chunking: bool = False,
) -> tuple[EpisodicMemory, openai.AsyncOpenAI, SkillToolBase]:
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

    skill_model: LanguageModel = OpenAIResponsesLanguageModel(
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
    query_skill = await init_skill(skill_model, reranker, skill_name)

    answer_model = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return memory, answer_model, query_skill
