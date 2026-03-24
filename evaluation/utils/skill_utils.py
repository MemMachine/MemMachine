import json
import os
import re
import time
from pathlib import Path
from typing import Any

import boto3
import neo4j
import openai
from memmachine_common import SkillRunner, install_skill
from memmachine_server.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine_server.common.metrics_factory import PrometheusMetricsFactory
from memmachine_server.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
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

RETRIEVE_SKILL_NAME = "RetrieveSkill"
SKILL_SPEC_ROOT = (
    Path(__file__).resolve().parents[2]
    / "packages/server/src/memmachine_server/retrieval_agent/agents/specs"
)
RETRIEVAL_HINT_UNCERTAINTY_PATTERN = re.compile(
    r"(?i)\b(if|likely|probably|suggests?|inferred?|assum(?:e|ed|ption)|"
    r"traditional|uncertain|unknown|not explicit|no explicit|may be|might)\b"
)


def _normalize_sub_agent_name(raw_name: str) -> str | None:
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


def _extract_sub_agents(perf_metrics: dict[str, Any]) -> list[str]:
    used_sub_agents: list[str] = []
    seen: set[str] = set()

    def _add(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        normalized_skill_name = _normalize_sub_agent_name(raw_name)
        if not normalized_skill_name:
            return
        if normalized_skill_name not in seen:
            used_sub_agents.append(normalized_skill_name)
            seen.add(normalized_skill_name)

    for run in perf_metrics.get("orchestrator_sub_agent_runs", []):
        if isinstance(run, dict):
            _add(run.get("agent_name"))
    _add(perf_metrics.get("selected_agent"))
    _add(perf_metrics.get("selected_agent_name"))
    return used_sub_agents


def _build_skill_used_label(perf_metrics: dict[str, Any]) -> str:
    labels: list[str] = []
    seen: set[str] = set()

    def _push(raw_name: object) -> None:
        if not isinstance(raw_name, str):
            return
        label = _normalize_sub_agent_name(raw_name) or raw_name.strip()
        if not label:
            return
        if label not in seen:
            labels.append(label)
            seen.add(label)

    # Keep top-level first, then append sub-skills in discovered order.
    _push(perf_metrics.get("skill"))
    for sub_agent in _extract_sub_agents(perf_metrics):
        _push(sub_agent)

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
    for run in perf_metrics.get("orchestrator_sub_agent_runs", []):
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


def _looks_like_non_answer(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized.endswith("?"):
        return True
    return normalized.startswith(
        (
            "i ",
            "there ",
            "this ",
            "that ",
            "which ",
            "do ",
            "does ",
            "did ",
            "it is unclear",
            "it is unknown",
        )
    )


def _response_usage_tokens(response: object) -> tuple[int, int]:
    usage = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage", None)
    )
    if usage is None:
        return 0, 0
    usage_payload = usage if isinstance(usage, dict) else {}
    input_tokens = usage_payload.get(
        "input_tokens",
        getattr(usage, "input_tokens", 0),
    )
    output_tokens = usage_payload.get(
        "output_tokens",
        getattr(usage, "output_tokens", 0),
    )
    return int(input_tokens or 0), int(output_tokens or 0)


def _count_runner_retrieved_episodes(search_results: list[dict[str, object]]) -> int:
    episode_keys: set[str] = set()
    fallback_counts: list[int] = []
    for payload in search_results:
        raw_episodes = payload.get("episodes")
        if isinstance(raw_episodes, list):
            for raw_episode in raw_episodes:
                if not isinstance(raw_episode, dict):
                    continue
                uid = raw_episode.get("uid")
                if isinstance(uid, str) and uid:
                    episode_keys.add(f"uid:{uid}")
                    continue
                content = raw_episode.get("content")
                if isinstance(content, str) and content.strip():
                    episode_keys.add(f"content:{content.strip()}")
                    continue
                episode_keys.add(
                    f"episode:{json.dumps(raw_episode, sort_keys=True, default=str)}"
                )

        count = payload.get("count")
        if isinstance(count, int | float) and not isinstance(count, bool):
            fallback_counts.append(int(count))

    if episode_keys:
        return len(episode_keys)
    return max(fallback_counts, default=0)


async def process_question_with_runner(  # noqa: C901
    answer_prompt: str,
    runner: SkillRunner | None,
    model: openai.AsyncOpenAI,
    question: str,
    answer: str,
    category: int | str,
    supporting_facts: list[str],
    adversarial_answer: str = "",
    model_name: str = "gpt-5-mini",
    full_content: str | None = None,
    extra_attributes: dict[str, Any] | None = None,
):
    total_input_tokens = 0
    total_output_tokens = 0
    llm_answering_time = 0.0
    retrieval_duration = 0.0

    if full_content is None:
        if runner is None:
            raise ValueError("runner is required when full_content is not provided.")
        runner = runner.fork()
        retrieval_start = time.perf_counter()
        runner_prompt = (
            _build_runner_question_prompt(
                answer_prompt=answer_prompt,
                question=question,
            )
            if getattr(runner, "use_answer_prompt_template", False)
            else question
        )
        response = await model.responses.create(
            model=model_name,
            max_output_tokens=4096,
            top_p=1,
            input=[{"role": "user", "content": runner.skill_messages(runner_prompt)}],
            tools=runner.tools(),
        )
        input_tokens, output_tokens = _response_usage_tokens(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        rsp_text = (await runner.handle_tool_loop(response)).strip()
        retrieval_end = time.perf_counter()
        retrieval_duration = retrieval_end - retrieval_start
        total_input_tokens += runner.last_follow_up_input_tokens
        total_output_tokens += runner.last_follow_up_output_tokens
        perf_metrics = _build_runner_perf_metrics(
            runner=runner,
            final_answer=rsp_text,
            retrieval_duration=retrieval_duration,
        )
    else:
        formatted_context = full_content
        prompt = answer_prompt.format(memories=formatted_context, question=question)
        answer_start = time.perf_counter()
        response = await model.responses.create(
            model=model_name,
            max_output_tokens=4096,
            top_p=1,
            input=[{"role": "user", "content": prompt}],
        )
        llm_answering_time += time.perf_counter() - answer_start
        input_tokens, output_tokens = _response_usage_tokens(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        rsp_text = response.output_text.strip()
        perf_metrics = {
            "memory_search_called": 0,
            "memory_retrieval_time": 0.0,
            "llm_time": 0.0,
            "memory_search_latency_seconds": [],
            "llm_call_count": 1,
            "skill": "PureLLM",
            "selected_skill_name": "PureLLM",
            "selected_agent_name": "PureLLM",
            "top_level_is_sufficient": False,
        }

    if full_content is not None:
        formatted_context = full_content
    else:
        stage_context = _format_runner_stage_context(
            runner.last_stage_results,
            runner.last_stage_sub_queries,
        )
        raw_context = _format_runner_context(runner.last_search_results)
        retrieval_answer_hint = _build_retrieval_answer_hint(perf_metrics)
        context_parts = [
            part
            for part in (
                "" if stage_context else retrieval_answer_hint,
                stage_context,
                raw_context,
            )
            if part
        ]
        formatted_context = "\n".join(context_parts).strip()

    if full_content is None:
        perf_metrics = _build_runner_perf_metrics(
            runner=runner,
            final_answer=rsp_text,
            retrieval_duration=retrieval_duration,
        )

    perf_metrics["llm_time"] = float(perf_metrics.get("llm_time", 0.0)) + llm_answering_time
    perf_metrics["llm_call_count"] = _extract_llm_call_count(perf_metrics)

    num_episodes_retrieved = _count_runner_retrieved_episodes(
        runner.last_search_results if runner is not None else []
    )
    memory_latency_breakdown = _extract_memory_search_latency_breakdown(perf_metrics)
    llm_call_count = _extract_llm_call_count(perf_metrics)
    mem_retrieval_time = perf_metrics.get("memory_retrieval_time", 0)
    llm_time = perf_metrics.get("llm_time", 0)
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
        f"LLM total time: {llm_time:.2f} seconds\n"
        f"LLM answering time: {llm_answering_time:.2f} seconds\n"
    )

    res = {
        "question": question,
        "golden_answer": answer,
        "model_answer": rsp_text,
        "category": category,
        "supporting_facts": supporting_facts,
        "adversarial_answer": adversarial_answer,
        "conversation_memories": formatted_context,
        "num_episodes_retrieved": num_episodes_retrieved,
        "llm_call_count": llm_call_count,
        "input_token": total_input_tokens,
        "output_token": total_output_tokens,
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


def _format_runner_context(search_results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for payload in search_results:
        text = payload.get("episodes_text")
        if not isinstance(text, str) or not text.strip():
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            dedup_key = re.sub(r"^\d+\.\s*", "", line)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            lines.append(dedup_key)
    return "\n".join(
        f"{index}. {line}" for index, line in enumerate(lines, start=1)
    ).strip()


def _build_runner_question_prompt(*, answer_prompt: str, question: str) -> str:
    return answer_prompt.format(
        memories=(
            "Use the memmachine_search tool to retrieve relevant memories. "
            "Tool outputs may include raw memory snippets plus [StageResult ...] and "
            "[SubQuery ...] lines from prior hops."
        ),
        question=question,
    )


def _answers_roughly_align(left: str, right: str) -> bool:
    normalized_left = _normalize_for_match(left)
    normalized_right = _normalize_for_match(right)
    if not normalized_left or not normalized_right:
        return False
    if (
        normalized_left == normalized_right
        or normalized_left in normalized_right
        or normalized_right in normalized_left
    ):
        return True
    left_tokens = {
        token for token in re.findall(r"[a-z0-9]+", normalized_left) if len(token) >= 2
    }
    right_tokens = {
        token for token in re.findall(r"[a-z0-9]+", normalized_right) if len(token) >= 2
    }
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap > 0 and (overlap / min(len(left_tokens), len(right_tokens))) >= 0.8


def _format_runner_stage_context(
    stage_results: list[dict[str, object]],
    stage_sub_queries: list[str],
) -> str:
    lines: list[str] = []
    for index, item in enumerate(stage_results, start=1):
        stage_query = str(item.get("query") or "").strip()
        stage_result = str(item.get("stage_result") or "").strip()
        if not stage_query or not stage_result:
            continue
        confidence = item.get("confidence_score")
        normalized_confidence = (
            float(confidence)
            if isinstance(confidence, int | float) and not isinstance(confidence, bool)
            else None
        )
        reason_note = item.get("reason_note")
        normalized_reason = (
            reason_note.strip()
            if isinstance(reason_note, str) and reason_note.strip()
            else None
        )
        reliability = _retrieval_hint_reliability(
            normalized_confidence,
            normalized_reason,
        )
        line = (
            f"[StageResult {index}] Query: {stage_query} | Answer: {stage_result} | "
            f"reliability={reliability}"
        )
        if normalized_confidence is not None:
            line += f" | Confidence: {normalized_confidence:.2f}"
        if normalized_reason is not None:
            line += f" | Reason: {normalized_reason}"
        lines.append(line)
    for index, sub_query in enumerate(stage_sub_queries, start=1):
        normalized_sub_query = sub_query.strip()
        if normalized_sub_query:
            lines.append(f"[SubQuery {index}] {normalized_sub_query}")
    return "\n".join(lines).strip()


def _build_runner_perf_metrics(  # noqa: C901
    *,
    runner: SkillRunner,
    final_answer: str,
    retrieval_duration: float,
) -> dict[str, Any]:
    perf_metrics: dict[str, Any] = {
        "memory_search_called": runner.last_memory_search_called,
        "memory_retrieval_time": retrieval_duration,
        "llm_time": retrieval_duration,
        "memory_search_latency_seconds": list(runner.last_memory_search_latency_seconds),
        "llm_call_count": runner.last_llm_call_count,
        "skill": RETRIEVE_SKILL_NAME,
        "selected_skill_name": "SkillRunner",
        "selected_agent_name": "SkillRunner",
        "top_level_is_sufficient": False,
    }

    stage_results: list[dict[str, object]] = []
    for item in runner.last_stage_results:
        query = item.get("query")
        stage_result = item.get("stage_result")
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(stage_result, str) or not stage_result.strip():
            continue
        normalized_item: dict[str, object] = {
            "query": query.strip(),
            "stage_result": stage_result.strip(),
        }
        confidence = item.get("confidence_score")
        if isinstance(confidence, int | float) and not isinstance(confidence, bool):
            normalized_item["confidence_score"] = float(confidence)
        reason_note = item.get("reason_note")
        if isinstance(reason_note, str) and reason_note.strip():
            normalized_item["reason_note"] = reason_note.strip()
        stage_results.append(normalized_item)

    if not stage_results:
        return perf_metrics

    latest_stage_result = stage_results[-1]
    latest_candidate = latest_stage_result["stage_result"]
    perf_metrics["latest_sufficiency_signal"] = True
    perf_metrics["latest_answer_candidate"] = latest_candidate

    latest_confidence = latest_stage_result.get("confidence_score")
    normalized_confidence = (
        float(latest_confidence)
        if isinstance(latest_confidence, int | float)
        and not isinstance(latest_confidence, bool)
        else None
    )
    if normalized_confidence is not None:
        perf_metrics["latest_sufficiency_confidence_score"] = normalized_confidence

    latest_reason = latest_stage_result.get("reason_note")
    if isinstance(latest_reason, str) and latest_reason.strip():
        perf_metrics["latest_sufficiency_reason_note"] = latest_reason.strip()

    if (
        normalized_confidence is not None
        and normalized_confidence >= runner.stage_result_confidence_threshold
    ):
        perf_metrics["answer_candidate"] = latest_candidate
        perf_metrics["top_level_confidence_score"] = normalized_confidence
        if isinstance(latest_reason, str) and latest_reason.strip():
            perf_metrics["top_level_reason_note"] = latest_reason.strip()
        if not _looks_like_non_answer(final_answer):
            perf_metrics["top_level_is_sufficient"] = _answers_roughly_align(
                latest_candidate,
                final_answer,
            )

    return perf_metrics


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
            or response.get("selected_agent_name")
            or response.get("selected_agent")
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
    message_sentence_chunking: bool = False,
    runner_config: dict | None = None,
    build_runner: bool = True,
) -> tuple[EpisodicMemory, openai.AsyncOpenAI, SkillRunner | None]:
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

    answer_model = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    if not build_runner:
        return memory, answer_model, None

    installed_skill = await install_skill(
        SKILL_SPEC_ROOT,
        "openai",
        openai_client=answer_model,
        skill_name="retrieve-skill",
    )
    return (
        memory,
        answer_model,
        SkillRunner(
            installed_skill,
            client=answer_model,
            model=model_name,
            search_mode="direct",
            direct_memory=memory,
            **(runner_config or {}),
        ),
    )
