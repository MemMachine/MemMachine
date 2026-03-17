from memmachine_server.common.episode_store import EpisodeEntry


def build_episode_entries(
    *,
    session_id: str,
    speaker: str,
    content: str,
    timestamp: str,
) -> list[EpisodeEntry]:
    entry = EpisodeEntry(
        content=content,
        producer_id=speaker,
        producer_role="user",
        created_at=timestamp,
        metadata={
            "locomo_session_id": session_id,
            "source_timestamp": timestamp,
            "source_speaker": speaker,
        },
    )
    return [entry]
