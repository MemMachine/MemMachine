"""Tests for reading a Claude Code transcript into event bodies."""

import json
from uuid import UUID, uuid5

from memmachine_client import claude_code_transcript

SESSION = "11111111-2222-3333-4444-555555555555"
PROPERTIES = {"agent": "claude-code", "project": "/repo"}


def write_transcript(path, entries):
    """Write entries as one JSON object per line, the way Claude Code does."""
    path.write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in entries), encoding="utf-8"
    )
    return path


def read_all(path, start_offset=0, start_index=0, session=SESSION):
    """Every event of a transcript, with the mark the reader ends on."""
    events = []
    mark = (start_offset, start_index)
    for produced, offset, index in claude_code_transcript.read_entries(
        path,
        session_id=session,
        properties=PROPERTIES,
        start_offset=start_offset,
        start_index=start_index,
    ):
        events.extend(produced)
        mark = (offset, index)
    return (events, *mark)


def user_entry(uuid, text, **fields):
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": SESSION,
        "timestamp": "2026-09-17T10:00:00.000Z",
        "message": {"role": "user", "content": text},
        **fields,
    }


def assistant_entry(uuid, content, **fields):
    return {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": SESSION,
        "timestamp": "2026-09-17T10:00:01.000Z",
        "message": {"role": "assistant", "content": content},
        **fields,
    }


def test_a_user_message_becomes_one_text_event(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [user_entry("aaaaaaaa-0000-4000-8000-000000000001", "why the backup?")],
    )

    events, offset, index = read_all(path)

    assert events == [
        {
            "id": "aaaaaaaa-0000-4000-8000-000000000001",
            "timestamp": "2026-09-17T10:00:00+00:00",
            "session_id": SESSION,
            "source_id": "claude-code",
            "context": {"author": {"name": "user"}},
            "blocks": [{"kind": "text", "text": "why the backup?"}],
            "properties": {"agent": "claude-code", "project": "/repo"},
        }
    ]
    assert (offset, index) == (path.stat().st_size, 1)


def test_an_assistant_turn_becomes_its_thinking_message_and_tool_call(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000002",
                [
                    {
                        "type": "thinking",
                        "thinking": "The install replaces the file.",
                        "signature": "…",
                    },
                    {"type": "text", "text": "Because the install replaces it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "/a/b.py"},
                    },
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    # Everything the entry holds, in the order it happened.
    assert [event["blocks"][0] for event in events] == [
        {"kind": "thinking", "text": "The install replaces the file."},
        {"kind": "text", "text": "Because the install replaces it."},
        {"kind": "tool_call", "name": "Read", "input": {"file_path": "/a/b.py"}},
    ]
    assert [event["context"] for event in events] == [
        {"author": {"name": "assistant"}},
        {"author": {"name": "assistant"}},
        {"author": {"name": "assistant"}},
    ]
    # The entry's own uuid holds its first event; the events after it in
    # the same entry are named under it.
    assert events[0]["id"] == "aaaaaaaa-0000-4000-8000-000000000002"
    assert events[1]["id"] == str(uuid5(UUID(events[0]["id"]), "1"))
    assert events[2]["id"] == str(uuid5(UUID(events[0]["id"]), "2"))
    assert events[2]["properties"]["tool_name"] == "Read"


def test_a_tool_result_is_named_after_the_call_it_answers(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000003",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000004",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "no such file",
                        "is_error": True,
                    }
                ],
            ),
        ],
    )

    events, _, _ = read_all(path)

    assert events[1]["blocks"] == [
        {
            "kind": "tool_result",
            "name": "Bash",
            "output": "no such file",
            "error": True,
        }
    ]
    assert events[1]["properties"]["tool_name"] == "Bash"
    assert events[1]["context"] == {"author": {"name": "user"}}


def test_a_result_read_after_its_call_is_still_named(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000005",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_3",
                        "name": "Grep",
                        "input": {"pattern": "x"},
                    }
                ],
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000006",
                [{"type": "tool_result", "tool_use_id": "toolu_3", "content": "hit"}],
            ),
        ],
    )
    first_line_length = len(path.read_text().splitlines(keepends=True)[0])

    events, _, _ = read_all(path, start_offset=first_line_length, start_index=1)

    # The call is before the mark, so the reader goes back for its name.
    assert events[0]["blocks"][0]["name"] == "Grep"
    assert events[0]["blocks"][0]["error"] is False


def test_a_result_of_blocks_keeps_its_text_and_names_the_rest(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000007",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_4",
                        "content": [
                            {"type": "text", "text": "one"},
                            {"type": "image", "source": {"data": "not-copied"}},
                        ],
                    }
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0]["output"] == "one\n[image]"


def test_injected_text_carries_where_it_came_from(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry("aaaaaaaa-0000-4000-8000-000000000011", "<system-reminder>go"),
            user_entry("aaaaaaaa-0000-4000-8000-000000000012", "<command-name>/loop"),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000013",
                "UserPromptSubmit hook additional context: a note",
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000014",
                "The following skills were invoked: design",
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000015",
                "This session is being continued from an earlier one",
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000016", "a skill body", isMeta=True
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000017",
                "the summary",
                isCompactSummary=True,
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000018",
                "I read <system-reminder> in the docs",
            ),
        ],
    )

    events, _, _ = read_all(path)

    assert [
        (event["blocks"][0]["kind"], event["blocks"][0].get("source"))
        for event in events
    ] == [
        ("injected", "reminder"),
        ("injected", "command"),
        ("injected", "hook"),
        ("injected", "skill"),
        ("injected", "compaction"),
        ("injected", "other"),
        ("injected", "compaction"),
        ("text", None),
    ]


def test_a_subagent_runs_in_a_session_of_its_own(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000021",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_5",
                        "name": "Agent",
                        "input": {"prompt": "look"},
                    }
                ],
            ),
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000022", "look at it", isSidechain=True
            ),
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000023",
                [{"type": "text", "text": "looked"}],
                isSidechain=True,
            ),
            user_entry("aaaaaaaa-0000-4000-8000-000000000024", "thanks"),
        ],
    )

    events, _, _ = read_all(path)

    parent, first, second, after = events
    assert parent["session_id"] == SESSION
    assert "parent_session" not in parent["properties"]
    assert first["session_id"] == second["session_id"] != SESSION
    assert first["properties"]["parent_session"] == SESSION
    assert after["session_id"] == SESSION


def test_a_subagent_that_names_itself_is_that_session(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000031",
                "run it",
                isSidechain=True,
                agentId="99999999-0000-4000-8000-000000000001",
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["session_id"] == "99999999-0000-4000-8000-000000000001"
    assert events[0]["properties"]["parent_session"] == SESSION


def test_reading_from_the_mark_reads_only_what_is_new(tmp_path):
    path = tmp_path / "session.jsonl"
    write_transcript(
        path, [user_entry("aaaaaaaa-0000-4000-8000-000000000041", "first")]
    )

    first_events, offset, index = read_all(path)
    path.write_text(
        path.read_text()
        + json.dumps(user_entry("aaaaaaaa-0000-4000-8000-000000000042", "second"))
        + "\n",
        encoding="utf-8",
    )
    second_events, _, _ = read_all(path, start_offset=offset, start_index=index)
    whole_again, _, _ = read_all(path)

    assert [event["id"] for event in first_events] == [
        "aaaaaaaa-0000-4000-8000-000000000041"
    ]
    assert [event["id"] for event in second_events] == [
        "aaaaaaaa-0000-4000-8000-000000000042"
    ]
    # Reading the whole file again produces the same events, so a lost
    # mark repeats a batch rather than writing anything twice.
    assert whole_again == first_events + second_events


def test_a_line_the_writer_has_not_finished_is_left_for_the_next_read(tmp_path):
    path = tmp_path / "session.jsonl"
    write_transcript(
        path, [user_entry("aaaaaaaa-0000-4000-8000-000000000051", "finished")]
    )
    finished_length = path.stat().st_size
    torn = json.dumps(user_entry("aaaaaaaa-0000-4000-8000-000000000052", "torn"))[:40]
    path.write_text(path.read_text() + torn, encoding="utf-8")

    events, offset, index = read_all(path)

    assert [event["id"] for event in events] == ["aaaaaaaa-0000-4000-8000-000000000051"]
    assert (offset, index) == (finished_length, 1)


def test_entries_that_carry_nothing_still_move_the_mark(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            {"type": "system", "uuid": "aaaaaaaa-0000-4000-8000-000000000061"},
            {"type": "file-history-snapshot", "uuid": "x"},
            user_entry("aaaaaaaa-0000-4000-8000-000000000062", "   "),
        ],
    )

    events, offset, index = read_all(path)

    assert events == []
    assert (offset, index) == (path.stat().st_size, 3)


def test_a_line_that_is_not_an_entry_is_passed_over(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        "not json\n"
        "[1, 2]\n"
        f"{json.dumps(user_entry('aaaaaaaa-0000-4000-8000-000000000071', 'read me'))}\n",
        encoding="utf-8",
    )

    events, offset, _ = read_all(path)

    assert [event["blocks"][0]["text"] for event in events] == ["read me"]
    assert offset == path.stat().st_size


def test_an_entry_without_a_time_is_left_for_the_server_to_stamp(tmp_path):
    entry = user_entry("aaaaaaaa-0000-4000-8000-000000000081", "no time")
    del entry["timestamp"]
    path = write_transcript(tmp_path / "session.jsonl", [entry])

    events, _, _ = read_all(path)

    assert "timestamp" not in events[0]


def test_a_result_whose_call_is_in_no_entry_is_named_unknown(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000091",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_gone",
                        "content": "answered",
                    }
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    # A block names a tool in one word or another, and the server holds
    # no block that names none.
    assert events[0]["blocks"][0]["name"] == "unknown"
    assert events[0]["properties"]["tool_name"] == "unknown"


def test_a_long_tool_result_is_cut_and_says_where(tmp_path):
    output = "x" * 9000
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000101",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_9",
                        "content": output,
                    }
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    written = events[0]["blocks"][0]["output"]
    assert written == f"{'x' * 8192}\n[truncated: 8192 of 9000 bytes]"
    assert len(written.encode()) == 8192 + len("\n[truncated: 8192 of 9000 bytes]")


def test_a_long_injected_passage_is_cut_and_a_message_is_not(tmp_path):
    long_text = "y" * 9000
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000102", f"<system-reminder>{long_text}"
            ),
            user_entry("aaaaaaaa-0000-4000-8000-000000000103", long_text),
        ],
    )

    events, _, _ = read_all(path)

    injected, message = events
    assert injected["blocks"][0]["text"].endswith("[truncated: 8192 of 9017 bytes]")
    # A message is what a query matches, and is written as it was.
    assert message["blocks"][0]["text"] == long_text


def test_a_cut_falls_on_a_character_boundary(tmp_path):
    # Three bytes each, so the cap falls inside the 2731st character.
    output = "é" * 5000
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            user_entry(
                "aaaaaaaa-0000-4000-8000-000000000104",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_10",
                        "content": output,
                    }
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    written, marker = events[0]["blocks"][0]["output"].rsplit("\n", 1)
    assert written == "é" * 4096
    assert marker == "[truncated: 8192 of 10000 bytes]"


def test_thinking_that_carries_nothing_is_no_event(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000111",
                [
                    {"type": "thinking", "thinking": "   ", "signature": "…"},
                    {"type": "text", "text": "done"},
                ],
            )
        ],
    )

    events, _, _ = read_all(path)

    assert [event["blocks"][0]["kind"] for event in events] == ["text"]


def test_long_thinking_is_cut_like_an_injected_passage(tmp_path):
    path = write_transcript(
        tmp_path / "session.jsonl",
        [
            assistant_entry(
                "aaaaaaaa-0000-4000-8000-000000000112",
                [{"type": "thinking", "thinking": "z" * 9000, "signature": "…"}],
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0] == {
        "kind": "thinking",
        "text": f"{'z' * 8192}\n[truncated: 8192 of 9000 bytes]",
    }
