"""Shared tool-loop helpers for installed skills and server agents."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, cast

_UNCERTAINTY_PATTERN = re.compile(
    r"(?i)\b(if|likely|probably|suggests?|inferred?|assum(?:e|ed|ption)|"
    r"traditional|uncertain|unknown|not explicit|no explicit|may be|might)\b"
)
_STAGE_RESULT_LINE_PATTERN = re.compile(
    r"^\[StageResult(?:\s+\d+)?\]\s*Query:\s*(?P<query>.+?)\s*\|\s*"
    r"Answer:\s*(?P<answer>.+?)"
    r"(?:\s*\|\s*Confidence:\s*(?P<confidence>[01](?:\.\d+)?))?"
    r"(?:\s*\|\s*Reason:\s*(?P<reason>.+))?$"
)
_SUBQUERY_LINE_PATTERN = re.compile(r"^\[SubQuery(?:\s+\d+)?\]\s*(?P<query>.+)$")
_QUERY_WHITESPACE_PATTERN = re.compile(r"\s+")
_RELATION_ATTRIBUTE_QUERY_PATTERN = re.compile(
    r"^(?P<entity>.+?)\s+"
    r"(?P<relation>father|mother|husband|wife|spouse|son|daughter|child|parent|brother|sister)\s+"
    r"(?P<attribute>born where|died when|death|death date|death cause|cause of death)$",
    re.IGNORECASE,
)
_BORN_WHERE_QUERY_PATTERN = re.compile(
    r"^(?P<entity>.+?)\s+born where(?:\s+born)?$",
    re.IGNORECASE,
)
_DIED_WHEN_QUERY_PATTERN = re.compile(
    r"^(?P<entity>.+?)\s+died when$",
    re.IGNORECASE,
)
_CAUSE_OF_DEATH_QUERY_PATTERN = re.compile(
    r"^(?P<entity>.+?)\s+(?:death cause|cause of death)(?:\s+.+)?$",
    re.IGNORECASE,
)
_CAUSE_OF_DEATH_PREFIX_QUERY_PATTERN = re.compile(
    r"^(?:what was the\s+)?cause of death(?:\s+of)?\s+(?P<entity>.+?)$",
    re.IGNORECASE,
)

STAGE_RESULT_GUIDANCE = (
    "If you continue after a memory search, first emit one compact line exactly as "
    "[StageResult] Query: <resolved hop> | Answer: <best current candidate> | "
    "Confidence: <0.00-1.00> | Reason: <short basis>. "
    "If another search is needed, also emit one line exactly as "
    "[SubQuery] <next query>. Keep both lines short. "
    "When you are finished, give the final answer plainly with no stage tags."
)


class SkillLoopContractError(ValueError):
    """Raised when a provider loop emits an unsupported tool contract."""


@dataclass(slots=True)
class SkillLoopToolCall:
    """Normalized tool-call payload emitted by a provider response."""

    name: str
    query: str
    call_id: str | None = None
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SkillLoopToolResult:
    """Tool execution result returned to a provider response continuation."""

    name: str
    output: dict[str, object]
    call_id: str | None = None
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SkillLoopState:
    """Mutable shared state for one provider tool loop."""

    llm_call_count: int = 1
    tool_call_count: int = 0
    follow_up_input_tokens: int = 0
    follow_up_output_tokens: int = 0
    memory_search_called: int = 0
    stage_results: list[dict[str, object]] = field(default_factory=list)
    stage_sub_queries: list[str] = field(default_factory=list)


class SkillLoopTransport(Protocol):
    """Provider-specific continuation hooks used by the shared loop."""

    def response_text(self, response: object) -> str:
        """Extract the latest assistant-visible text from a provider response."""

    def tool_calls(self, response: object) -> list[SkillLoopToolCall]:
        """Extract normalized tool-call records from a provider response."""

    async def continue_with_results(
        self,
        *,
        response: object,
        tool_results: list[SkillLoopToolResult],
    ) -> object:
        """Submit tool outputs and return the next provider response."""

    def usage(self, response: object) -> tuple[int, int]:
        """Return input/output token usage for one provider response."""


ToolSearchRunner = Callable[[str, SkillLoopState], Awaitable[dict[str, object]]]


def augment_prompt(
    prompt: str,
    *,
    stage_result_mode: bool,
) -> str:
    """Append stage-result guidance to a user prompt when enabled."""
    if not stage_result_mode:
        return prompt
    return f"{prompt}\n\n{STAGE_RESULT_GUIDANCE}"


def as_dict(raw_value: object) -> dict[str, object]:
    """Normalize supported object-like payloads into plain dicts."""
    if isinstance(raw_value, dict):
        return {str(key): value for key, value in raw_value.items()}
    model_dump = getattr(raw_value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return {str(key): value for key, value in dumped.items()}
    raw_dict = getattr(raw_value, "__dict__", None)
    if isinstance(raw_dict, dict):
        return {str(key): value for key, value in raw_dict.items()}
    return {}


def as_list(raw_value: object) -> list[object]:
    """Normalize supported list payloads into plain object lists."""
    return cast(list[object], raw_value) if isinstance(raw_value, list) else []


def extract_query_from_arguments(raw_arguments: object) -> str:
    """Best-effort extraction of a `query` field from tool arguments."""
    arguments = normalize_tool_arguments(raw_arguments)
    if arguments:
        query = arguments.get("query")
        return canonicalize_search_query(query) if isinstance(query, str) else ""
    return ""


def canonicalize_search_query(query: str) -> str:
    """Normalize common malformed search phrasings into stable retrieval queries."""
    normalized = _QUERY_WHITESPACE_PATTERN.sub(" ", query).strip().rstrip("?").strip()
    if not normalized:
        return ""

    relation_match = _RELATION_ATTRIBUTE_QUERY_PATTERN.match(normalized)
    if relation_match:
        entity = relation_match.group("entity").strip()
        relation = relation_match.group("relation").strip().lower()
        return f"{entity} {relation}"

    born_match = _BORN_WHERE_QUERY_PATTERN.match(normalized)
    if born_match:
        entity = born_match.group("entity").strip()
        return f"Where was {entity} born?"

    died_match = _DIED_WHEN_QUERY_PATTERN.match(normalized)
    if died_match:
        entity = died_match.group("entity").strip()
        return f"When did {entity} die?"

    death_prefix_match = _CAUSE_OF_DEATH_PREFIX_QUERY_PATTERN.match(normalized)
    if death_prefix_match:
        entity = death_prefix_match.group("entity").strip()
        return f"What was the cause of death of {entity}?"

    death_match = _CAUSE_OF_DEATH_QUERY_PATTERN.match(normalized)
    if death_match:
        entity = death_match.group("entity").strip()
        return f"What was the cause of death of {entity}?"

    return normalized


def normalize_tool_arguments(raw_arguments: object) -> dict[str, object]:
    """Normalize tool arguments passed as either JSON text or object payloads."""
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw_arguments, dict):
        return {str(key): value for key, value in raw_arguments.items()}
    return {}


def normalize_search_result(
    *,
    query: str,
    raw_result: object,
    score_threshold: float | None = None,
    max_episode_chars: int | None = None,
    stage_result_mode: bool = False,
    stage_results: list[dict[str, object]] | None = None,
    stage_sub_queries: list[str] | None = None,
) -> dict[str, object]:
    """Normalize a search response into the compact payload used by tool loops."""
    payload = as_dict(raw_result)
    content_payload = as_dict(payload.get("content")) or payload
    episodic_payload = as_dict(
        content_payload.get("episodic_memory")
        if "episodic_memory" in content_payload
        else payload
    )
    episodic = as_dict(episodic_payload)
    short_term = as_dict(episodic.get("short_term_memory"))
    long_term = as_dict(episodic.get("long_term_memory"))
    semantic_memory = normalize_semantic_memory(
        content_payload.get("semantic_memory", payload.get("semantic_memory", []))
    )

    episode_summary = [
        item
        for item in as_list(short_term.get("episode_summary"))
        if isinstance(item, str) and item.strip()
    ]
    episodes = [
        *normalize_episodes(
            short_term.get("episodes", []),
            score_threshold=score_threshold,
        ),
        *normalize_episodes(
            long_term.get("episodes", []),
            score_threshold=score_threshold,
        ),
    ]
    episode_lines: list[str] = []
    for index, episode in enumerate(episodes, start=1):
        content = episode.get("content")
        if isinstance(content, str) and content.strip():
            episode_lines.append(
                f"{index}. {episode_content(content, max_episode_chars=max_episode_chars)}"
            )
    semantic_lines = [
        semantic_memory_line(feature)
        for feature in semantic_memory
        if semantic_memory_line(feature)
    ]
    stage_result_memory = stage_result_memory_lines(
        stage_result_mode=stage_result_mode,
        stage_results=stage_results or [],
        stage_sub_queries=stage_sub_queries or [],
    )
    episodic_memory: dict[str, object] = {
        "short_term_memory": {
            "episodes": normalize_episodes(
                short_term.get("episodes", []),
                score_threshold=score_threshold,
            ),
            "episode_summary": episode_summary,
        },
        "long_term_memory": {
            "episodes": normalize_episodes(
                long_term.get("episodes", []),
                score_threshold=score_threshold,
            ),
        },
    }
    combined_text = "\n".join(
        [*semantic_lines, *episode_summary, *episode_lines]
    ).strip()

    normalized_payload: dict[str, object] = {
        "query": query,
        "episodic_memory": episodic_memory,
        "semantic_memory": semantic_memory,
        "episode_summary": episode_summary,
        "episodes": episodes,
        "episodes_text": combined_text,
        "semantic_text": "\n".join(semantic_lines).strip(),
        "memory_text": combined_text,
        "count": len(episodes),
        "semantic_count": len(semantic_memory),
        "total_count": len(episodes) + len(semantic_memory),
    }
    if stage_result_memory:
        normalized_payload["stage_result_memory"] = stage_result_memory
        normalized_payload["stage_result_instructions"] = STAGE_RESULT_GUIDANCE
    return normalized_payload


def normalize_episodes(
    raw_episodes: object,
    *,
    score_threshold: float | None,
) -> list[dict[str, object]]:
    """Normalize episode arrays and apply an optional score filter."""
    if not isinstance(raw_episodes, list):
        return []
    normalized: list[dict[str, object]] = []
    for raw_episode in raw_episodes:
        episode = as_dict(raw_episode)
        if not episode:
            continue
        if score_threshold is not None:
            raw_score = episode.get("score", episode.get("relevance_score", 1.0))
            score = _coerce_float(raw_score, default=1.0)
            if score < score_threshold:
                continue
        normalized.append(episode)
    return normalized


def _coerce_float(value: object, *, default: float) -> float:
    """Best-effort float coercion for loosely typed provider payloads."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        with suppress(ValueError):
            return float(value)
    return default


def episode_content(content: str, *, max_episode_chars: int | None) -> str:
    """Apply optional truncation to an episode content string."""
    if max_episode_chars is None:
        return content
    return content[:max_episode_chars]


def normalize_semantic_memory(raw_features: object) -> list[dict[str, object]]:
    """Normalize semantic feature arrays into plain dicts."""
    if not isinstance(raw_features, list):
        return []
    normalized: list[dict[str, object]] = []
    for raw_feature in raw_features:
        feature = as_dict(raw_feature)
        if feature:
            normalized.append(feature)
    return normalized


def semantic_memory_line(feature: dict[str, object]) -> str:
    """Render one compact semantic-memory line for tool-loop context."""
    feature_name = str(feature.get("feature_name") or "").strip()
    value = str(feature.get("value") or "").strip()
    category = str(feature.get("category") or "").strip()
    tag = str(feature.get("tag") or "").strip()
    if feature_name and value:
        label = feature_name
    elif value:
        label = value
    elif feature_name:
        label = feature_name
    else:
        return ""
    prefix_parts = [part for part in (category, tag) if part]
    prefix = "/".join(prefix_parts)
    if prefix:
        return f"[Semantic] {prefix} | {label}: {value or feature_name}"
    return f"[Semantic] {label}: {value or feature_name}"


def stage_result_memory_lines(
    *,
    stage_result_mode: bool,
    stage_results: list[dict[str, object]],
    stage_sub_queries: list[str],
) -> list[str]:
    """Render stage-result and sub-query memory lines for later hops."""
    if not stage_result_mode:
        return []

    lines: list[str] = []
    for index, item in enumerate(stage_results, start=1):
        query = str(item.get("query") or "").strip()
        answer = str(item.get("stage_result") or "").strip()
        if not query or not answer:
            continue
        line = f"[StageResult {index}] Query: {query} | Answer: {answer}"
        confidence = item.get("confidence_score")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            line += f" | Confidence: {float(confidence):.2f}"
        reason = item.get("reason_note")
        if isinstance(reason, str) and reason.strip():
            line += f" | Reason: {reason.strip()}"
        lines.append(line)

    for index, subquery in enumerate(stage_sub_queries, start=1):
        lines.append(f"[SubQuery {index}] {subquery}")
    return lines


def should_omit_tool_episode_text(
    *,
    omit_episode_text_on_confident_stage_result: bool,
    stage_result_mode: bool,
    stage_results: list[dict[str, object]],
    stage_result_confidence_threshold: float,
) -> bool:
    """Return whether raw episode text can be omitted in favor of stage memory."""
    if not omit_episode_text_on_confident_stage_result:
        return False
    if not stage_result_mode or not stage_results:
        return False
    latest_stage_result = stage_results[-1]
    confidence = latest_stage_result.get("confidence_score")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return False
    return float(confidence) >= stage_result_confidence_threshold


def build_tool_search_result(
    *,
    payload: dict[str, object],
    stage_result_mode: bool,
    stage_results: list[dict[str, object]],
    omit_episode_text_on_confident_stage_result: bool,
    stage_result_confidence_threshold: float,
) -> dict[str, object]:
    """Build the compact tool payload returned back to the model."""
    compact_payload: dict[str, object] = {
        "query": payload.get("query", ""),
        "count": payload.get("count", 0),
        "semantic_count": payload.get("semantic_count", 0),
        "total_count": payload.get("total_count", payload.get("count", 0)),
    }
    episodic_memory = payload.get("episodic_memory")
    if isinstance(episodic_memory, dict) and episodic_memory:
        compact_payload["episodic_memory"] = episodic_memory
    semantic_memory = payload.get("semantic_memory")
    if isinstance(semantic_memory, list) and semantic_memory:
        compact_payload["semantic_memory"] = semantic_memory
    semantic_text = payload.get("semantic_text")
    if isinstance(semantic_text, str) and semantic_text.strip():
        compact_payload["semantic_text"] = semantic_text
    stage_result_memory = payload.get("stage_result_memory")
    if isinstance(stage_result_memory, list) and stage_result_memory:
        compact_payload["stage_result_memory"] = stage_result_memory
    if not should_omit_tool_episode_text(
        omit_episode_text_on_confident_stage_result=(
            omit_episode_text_on_confident_stage_result
        ),
        stage_result_mode=stage_result_mode,
        stage_results=stage_results,
        stage_result_confidence_threshold=stage_result_confidence_threshold,
    ):
        compact_payload["episodes_text"] = payload.get("episodes_text", "")
        compact_payload["memory_text"] = payload.get("memory_text", "")
    return compact_payload


def record_stage_progress(  # noqa: C901
    latest_text: str,
    *,
    stage_result_mode: bool,
    stage_results: list[dict[str, object]],
    stage_sub_queries: list[str],
) -> None:
    """Parse stage-result markers from assistant text into loop state."""
    if not stage_result_mode or not latest_text.strip():
        return

    for raw_line in latest_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stage_match = _STAGE_RESULT_LINE_PATTERN.match(line)
        if stage_match:
            query = canonicalize_search_query(stage_match.group("query"))
            answer = stage_match.group("answer").strip()
            if not query or not answer:
                continue
            record: dict[str, object] = {
                "query": query,
                "stage_result": answer,
            }
            confidence_raw = stage_match.group("confidence")
            if confidence_raw is not None:
                with suppress(ValueError):
                    record["confidence_score"] = float(confidence_raw)
            reason = stage_match.group("reason")
            if reason is not None and reason.strip():
                record["reason_note"] = reason.strip()
            key = (query, answer)
            existing = {
                (
                    str(item.get("query", "")),
                    str(item.get("stage_result", "")),
                )
                for item in stage_results
            }
            if key not in existing:
                stage_results.append(record)
            continue

        subquery_match = _SUBQUERY_LINE_PATTERN.match(line)
        if subquery_match:
            subquery = canonicalize_search_query(subquery_match.group("query"))
            if subquery and subquery not in stage_sub_queries:
                stage_sub_queries.append(subquery)


def final_response_text(latest_text: str, *, stage_result_mode: bool) -> str:
    """Strip stage-result lines from a final assistant response."""
    stripped = latest_text.strip()
    if not stage_result_mode or not stripped:
        return stripped
    clean_lines = [
        line
        for line in stripped.splitlines()
        if not _STAGE_RESULT_LINE_PATTERN.match(line.strip())
        and not _SUBQUERY_LINE_PATTERN.match(line.strip())
    ]
    cleaned = "\n".join(line for line in clean_lines if line.strip()).strip()
    return cleaned or stripped


async def continue_tool_loop(
    *,
    initial_response: object,
    transport: SkillLoopTransport,
    state: SkillLoopState,
    run_search: ToolSearchRunner,
    max_turns: int,
    stage_result_mode: bool,
    early_exit_confidence: bool,
    partial_response_text: str,
) -> str:
    """Run a provider-native memmachine_search loop from an initial response."""
    latest_text = transport.response_text(initial_response)
    record_stage_progress(
        latest_text,
        stage_result_mode=stage_result_mode,
        stage_results=state.stage_results,
        stage_sub_queries=state.stage_sub_queries,
    )
    response = initial_response

    for turn_index in range(max_turns):
        tool_calls = transport.tool_calls(response)
        state.tool_call_count += len(tool_calls)
        if not tool_calls:
            return final_response_text(latest_text, stage_result_mode=stage_result_mode)
        if turn_index + 1 >= max_turns:
            return (
                final_response_text(
                    latest_text,
                    stage_result_mode=stage_result_mode,
                )
                or partial_response_text
            )

        tool_results: list[SkillLoopToolResult] = []
        for call in tool_calls:
            if call.name != "memmachine_search":
                raise SkillLoopContractError(f"Unsupported tool name: {call.name}")
            tool_results.append(
                SkillLoopToolResult(
                    name=call.name,
                    call_id=call.call_id,
                    arguments=call.arguments,
                    output=await run_search(call.query, state),
                )
            )

        if (
            early_exit_confidence
            and state.memory_search_called >= 1
            and latest_text
            and not _UNCERTAINTY_PATTERN.search(latest_text)
        ):
            return latest_text.strip()

        response = await transport.continue_with_results(
            response=response,
            tool_results=tool_results,
        )
        state.llm_call_count += 1
        input_tokens, output_tokens = transport.usage(response)
        state.follow_up_input_tokens += input_tokens
        state.follow_up_output_tokens += output_tokens
        latest_text = transport.response_text(response)
        record_stage_progress(
            latest_text,
            stage_result_mode=stage_result_mode,
            stage_results=state.stage_results,
            stage_sub_queries=state.stage_sub_queries,
        )

    return final_response_text(latest_text, stage_result_mode=stage_result_mode)
