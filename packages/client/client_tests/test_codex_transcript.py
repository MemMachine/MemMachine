"""Tests for reading a Codex rollout file into event bodies."""

import json
from uuid import NAMESPACE_URL, UUID, uuid5

from memmachine_client import codex_transcript

SESSION = "019fd7f4-8e96-7732-9188-d4c64043f064"
PROPERTIES = {"agent": "codex", "project": "/repo"}


def write_rollout(path, records):
    """Write records as one JSON object per line, the way Codex does."""
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )
    return path


def read_all(path, start_offset=0, start_index=0, session=SESSION):
    """Every event of a rollout, with the mark the reader ends on."""
    events = []
    mark = (start_offset, start_index)
    for produced, offset, index in codex_transcript.read_entries(
        path,
        session_id=session,
        properties=PROPERTIES,
        start_offset=start_offset,
        start_index=start_index,
    ):
        events.extend(produced)
        mark = (offset, index)
    return (events, *mark)


def response_item(payload, timestamp="2026-09-17T10:00:00.000+00:00"):
    return {"timestamp": timestamp, "type": "response_item", "payload": payload}


def message(role, text):
    return response_item(
        {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": text,
                }
            ],
        }
    )


def test_a_user_message_becomes_one_text_event(tmp_path):
    path = write_rollout(tmp_path / "rollout.jsonl", [message("user", "add the hook")])

    events, offset, index = read_all(path)

    assert events == [
        {
            "id": str(uuid5(UUID(SESSION), "0")),
            "timestamp": "2026-09-17T10:00:00+00:00",
            "session_id": SESSION,
            "source_id": "codex",
            "context": {"author": {"name": "user"}},
            "blocks": [{"kind": "text", "text": "add the hook"}],
            "properties": {"agent": "codex", "project": "/repo"},
        }
    ]
    assert (offset, index) == (path.stat().st_size, 1)


def test_an_assistant_message_is_the_assistant_speaking(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl", [message("assistant", "the hook is written")]
    )

    events, _, _ = read_all(path)

    assert events[0]["context"] == {"author": {"name": "assistant"}}
    assert events[0]["blocks"] == [{"kind": "text", "text": "the hook is written"}]


def test_a_function_call_carries_the_arguments_it_was_given(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call_1",
                    "arguments": '{"command": ["ls", "-la"]}',
                }
            ),
            response_item(
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"output": "a.py\\n", "metadata": {"exit_code": 0}}',
                }
            ),
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"] == [
        {"kind": "tool_call", "name": "shell", "input": {"command": ["ls", "-la"]}}
    ]
    assert events[0]["properties"]["tool_name"] == "shell"
    assert events[1]["blocks"] == [
        {"kind": "tool_result", "name": "shell", "output": "a.py\n", "error": False}
    ]


def test_an_exit_code_of_its_own_makes_a_result_a_failure(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "custom_tool_call",
                    "name": "shell",
                    "call_id": "call_2",
                    "input": "ls /nowhere",
                }
            ),
            response_item(
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_2",
                    "output": '{"output": "no such file", "metadata": {"exit_code": 2}}',
                }
            ),
        ],
    )

    events, _, _ = read_all(path)

    # A custom tool call carries a body of its own rather than JSON.
    assert events[0]["blocks"][0]["input"] == {"input": "ls /nowhere"}
    assert events[1]["blocks"] == [
        {
            "kind": "tool_result",
            "name": "shell",
            "output": "no such file",
            "error": True,
        }
    ]


def test_a_result_of_blocks_and_a_result_of_plain_text(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "function_call_output",
                    "call_id": "call_3",
                    "output": [
                        {"type": "input_text", "text": "one"},
                        {"type": "input_text", "text": "two"},
                    ],
                }
            ),
            response_item(
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_4",
                    "output": "plain output",
                }
            ),
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0]["output"] == "one\ntwo"
    # No call in the file carries this id, so the tool is unnamed.
    assert events[0]["blocks"][0]["name"] == "unknown"
    assert events[1]["blocks"][0]["output"] == "plain output"


def test_injected_text_carries_where_it_came_from(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            message(
                "developer", "<skills_instructions>read them</skills_instructions>"
            ),
            message("developer", "<something_new>a channel of its own</something_new>"),
            message("user", "<environment_context>cwd is /repo</environment_context>"),
            message(
                "user", "<current_time_reminder>it is late</current_time_reminder>"
            ),
            message("user", "<user_shell_command>git status</user_shell_command>"),
            message("user", "what does <environment_context> hold?"),
        ],
    )

    events, _, _ = read_all(path)

    assert [
        (event["blocks"][0]["kind"], event["blocks"][0].get("source"))
        for event in events
    ] == [
        ("injected", "skill"),
        ("injected", "other"),
        ("injected", "other"),
        ("injected", "reminder"),
        ("injected", "command"),
        ("text", None),
    ]
    assert all(event["context"] == {"author": {"name": "user"}} for event in events)


def test_a_compaction_summary_is_injected_text(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            {
                "timestamp": "2026-09-17T10:00:00.000+00:00",
                "type": "compacted",
                "payload": {
                    "message": "what the session did",
                    "replacement_history": [],
                },
            }
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"] == [
        {"kind": "injected", "text": "what the session did", "source": "compaction"}
    ]


def test_the_records_that_restate_the_conversation_are_left_behind(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            {
                "timestamp": "2026-09-17T10:00:00.000+00:00",
                "type": "session_meta",
                "payload": {"id": SESSION, "cwd": "/repo"},
            },
            {
                "timestamp": "2026-09-17T10:00:01.000+00:00",
                "type": "turn_context",
                "payload": {"cwd": "/repo", "model": "gpt-5-codex"},
            },
            {
                "timestamp": "2026-09-17T10:00:02.000+00:00",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "said once already"},
            },
            # Every reasoning record written on this machine looks like
            # this one: nothing but an encrypted body, which is not text.
            response_item(
                {"type": "reasoning", "summary": [], "encrypted_content": "x"}
            ),
            message("assistant", "said once already"),
        ],
    )

    events, offset, index = read_all(path)

    assert [event["blocks"][0]["text"] for event in events] == ["said once already"]
    assert events[0]["id"] == str(uuid5(UUID(SESSION), "4"))
    assert (offset, index) == (path.stat().st_size, 5)


def test_reading_from_the_mark_reads_only_what_is_new(tmp_path):
    path = tmp_path / "rollout.jsonl"
    write_rollout(path, [message("user", "first")])

    first_events, offset, index = read_all(path)
    path.write_text(
        path.read_text() + json.dumps(message("user", "second")) + "\n",
        encoding="utf-8",
    )
    second_events, _, _ = read_all(path, start_offset=offset, start_index=index)
    whole_again, _, _ = read_all(path)

    assert [event["blocks"][0]["text"] for event in second_events] == ["second"]
    # The index the mark carries is what names a record, so reading the
    # whole file again produces the same events.
    assert whole_again == first_events + second_events


def test_a_session_that_is_not_a_uuid_still_names_its_events(tmp_path):
    path = write_rollout(tmp_path / "rollout.jsonl", [message("user", "hello")])

    events, _, _ = read_all(path, session="thread-42")

    assert events[0]["session_id"] == "thread-42"
    assert events[0]["id"] == str(uuid5(uuid5(NAMESPACE_URL, "thread-42"), "0"))


def test_a_line_the_writer_has_not_finished_is_left_for_the_next_read(tmp_path):
    path = tmp_path / "rollout.jsonl"
    write_rollout(path, [message("user", "finished")])
    finished_length = path.stat().st_size
    path.write_text(
        path.read_text() + json.dumps(message("user", "torn"))[:30], encoding="utf-8"
    )

    events, offset, index = read_all(path)

    assert [event["blocks"][0]["text"] for event in events] == ["finished"]
    assert (offset, index) == (finished_length, 1)


def test_a_long_tool_output_is_cut_and_says_where(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "function_call_output",
                    "call_id": "call_9",
                    "output": "x" * 9000,
                }
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0]["output"] == (
        f"{'x' * 8192}\n[truncated: 8192 of 9000 bytes]"
    )


def test_a_long_injected_passage_is_cut_and_a_message_is_not(tmp_path):
    long_text = "y" * 9000
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            message("developer", long_text),
            message("assistant", long_text),
        ],
    )

    events, _, _ = read_all(path)

    injected, spoken = events
    assert injected["blocks"][0]["text"].endswith("[truncated: 8192 of 9000 bytes]")
    assert spoken["blocks"][0]["text"] == long_text


def test_reasoning_that_carries_text_is_written_down(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "The hook is missing."}
                    ],
                    "content": None,
                    "encrypted_content": "opaque",
                }
            ),
            response_item(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": "Write it, then test it."}
                    ],
                    "encrypted_content": "opaque",
                }
            ),
        ],
    )

    events, _, _ = read_all(path)

    assert [event["blocks"][0] for event in events] == [
        {"kind": "thinking", "text": "The hook is missing."},
        {"kind": "thinking", "text": "Write it, then test it."},
    ]
    assert events[0]["context"] == {"author": {"name": "assistant"}}
    # The encrypted body is not text, and is never posted.
    assert "opaque" not in json.dumps(events)


def test_reasoning_with_a_summary_and_content_carries_both(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "First."}],
                    "content": [{"type": "reasoning_text", "text": "Then this."}],
                    "encrypted_content": "opaque",
                }
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0]["text"] == "First.\nThen this."


def test_long_thinking_is_cut_like_an_injected_passage(tmp_path):
    path = write_rollout(
        tmp_path / "rollout.jsonl",
        [
            response_item(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "z" * 9000}],
                    "encrypted_content": "opaque",
                }
            )
        ],
    )

    events, _, _ = read_all(path)

    assert events[0]["blocks"][0]["text"] == (
        f"{'z' * 8192}\n[truncated: 8192 of 9000 bytes]"
    )
