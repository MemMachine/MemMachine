from evaluation.semantic_memory.ingest import build_episode_entries


def test_build_episode_entries_sets_metadata():
    entries = build_episode_entries(
        session_id="session_1",
        speaker="Alice",
        content="Hello",
        timestamp="2024-01-01T00:00:00Z",
    )
    assert entries[0].metadata is not None
    assert entries[0].metadata["source_speaker"] == "Alice"
