"""Persistent session-state models for top-level retrieval-agent orchestration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from memmachine_server.common.episode_store import Episode


class AgentSessionEvent(BaseModel):
    """A single event emitted during top-level orchestration."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    actor: str
    event_type: str
    detail: str


class AgentToolCallRecord(BaseModel):
    """A single tool-call record captured in orchestrator session state."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    tool_name: str
    arguments: dict[str, object]
    status: str
    result_summary: str = ""
    raw_result: dict[str, object] | None = None


class SubAgentRunRecord(BaseModel):
    """A single spawned sub-agent execution record."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    agent_name: str
    query: str
    status: str
    fallback_trigger_reason: str | None = None
    tool_calls: list[AgentToolCallRecord] = Field(default_factory=list)
    llm_call_count: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_time: float = 0.0
    episodes_returned: int = 0
    branch_total: int = 0
    branch_success_count: int = 0
    branch_failure_count: int = 0
    branch_retry_count: int = 0
    normalization_warnings: list[str] = Field(default_factory=list)


class TopLevelAgentSessionState(BaseModel):
    """Persistent state owned by the top-level agent during one query lifecycle."""

    model_config = ConfigDict(extra="forbid")

    route_name: str
    policy_name: str
    query: str
    current_step: int = 0
    events: list[AgentSessionEvent] = Field(default_factory=list)
    tool_calls: list[AgentToolCallRecord] = Field(default_factory=list)
    sub_agent_runs: list[SubAgentRunRecord] = Field(default_factory=list)
    merged_episodes: list[Episode] = Field(default_factory=list)
    final_response: str | None = None
    completed: bool = False

    @classmethod
    def new(
        cls,
        *,
        route_name: str,
        policy_name: str,
        query: str,
    ) -> TopLevelAgentSessionState:
        """Create a fresh session-state object for top-level orchestration."""
        return cls(route_name=route_name, policy_name=policy_name, query=query)

    def next_step(self) -> int:
        """Advance and return the current orchestration step index."""
        self.current_step += 1
        return self.current_step

    def record_event(
        self,
        *,
        actor: str,
        event_type: str,
        detail: str,
    ) -> None:
        """Append an orchestration event to session history."""
        self.events.append(
            AgentSessionEvent(
                step=self.current_step,
                actor=actor,
                event_type=event_type,
                detail=detail,
            )
        )

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        status: str,
        result_summary: str = "",
        raw_result: dict[str, object] | None = None,
    ) -> AgentToolCallRecord:
        """Append a tool-call record and return it."""
        record = AgentToolCallRecord(
            step=self.current_step,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            result_summary=result_summary,
            raw_result=raw_result,
        )
        self.tool_calls.append(record)
        return record

    def record_sub_agent_run(
        self,
        *,
        agent_name: str,
        query: str,
        status: str,
        fallback_trigger_reason: str | None = None,
        tool_calls: list[AgentToolCallRecord] | None = None,
        llm_call_count: int = 0,
        llm_input_tokens: int = 0,
        llm_output_tokens: int = 0,
        llm_time: float = 0.0,
        episodes_returned: int = 0,
        branch_total: int = 0,
        branch_success_count: int = 0,
        branch_failure_count: int = 0,
        branch_retry_count: int = 0,
        normalization_warnings: list[str] | None = None,
    ) -> None:
        """Append a sub-agent execution record."""
        self.sub_agent_runs.append(
            SubAgentRunRecord(
                step=self.current_step,
                agent_name=agent_name,
                query=query,
                status=status,
                fallback_trigger_reason=fallback_trigger_reason,
                tool_calls=tool_calls or [],
                llm_call_count=llm_call_count,
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
                llm_time=llm_time,
                episodes_returned=episodes_returned,
                branch_total=branch_total,
                branch_success_count=branch_success_count,
                branch_failure_count=branch_failure_count,
                branch_retry_count=branch_retry_count,
                normalization_warnings=normalization_warnings or [],
            )
        )

    def merge_episodes(self, episodes: list[Episode]) -> None:
        """Merge episodes into state while preserving first-seen uid order."""
        existing_uids = {episode.uid for episode in self.merged_episodes}
        for episode in episodes:
            if episode.uid in existing_uids:
                continue
            self.merged_episodes.append(episode)
            existing_uids.add(episode.uid)

    def finalize(self, *, response: str | None = None) -> None:
        """Mark session as complete and optionally capture final response text."""
        if response is not None:
            self.final_response = response
        self.completed = True

    def prompt_snapshot(self) -> str:
        """Render compact state summary for model prompts."""
        agent_names = [run.agent_name for run in self.sub_agent_runs]
        episode_uids = [episode.uid for episode in self.merged_episodes]
        return (
            f"step={self.current_step}; "
            f"events={len(self.events)}; "
            f"tool_calls={len(self.tool_calls)}; "
            f"sub_agents={agent_names}; "
            f"episodes={episode_uids}; "
            f"completed={self.completed}"
        )

    def full_trace_snapshot(self) -> dict[str, object]:
        """Return full per-step/per-tool trace payload for metrics surfaces."""
        return {
            "events": [event.model_dump(mode="json") for event in self.events],
            "tool_calls": [call.model_dump(mode="json") for call in self.tool_calls],
            "sub_agent_runs": [
                run.model_dump(mode="json") for run in self.sub_agent_runs
            ],
        }
