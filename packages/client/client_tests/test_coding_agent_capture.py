"""Tests for the capture client the agents' `Stop` hooks run."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

from memmachine_client import cli, coding_agent_capture

SESSION = "11111111-2222-3333-4444-555555555555"
TENANT = "alice"


class EventsHandler(BaseHTTPRequestHandler):
    """Answers the batches a capture posts, out of what a test scripted."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        server = cast("EventsServer", self.server)
        length = int(self.headers.get("Content-Length", "0"))
        events = json.loads(self.rfile.read(length) or b"[]")
        server.paths.append(self.path)
        server.batches.append(events)
        time.sleep(server.delay)
        status, body = (
            server.answers.pop(0)
            if server.answers
            else (200, {"stored": [event["id"] for event in events]})
        )
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_request(self, code="-", size="-"):
        """Keep the server's own access log out of the test output."""


class EventsServer(ThreadingHTTPServer):
    """A server standing in for a MemMachine server's events route."""

    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), EventsHandler)
        self.paths: list[str] = []
        self.batches: list[list[dict]] = []
        self.answers: list[tuple[int, dict]] = []
        self.delay = 0.0

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"

    def handle_error(self, request, client_address):
        """Keep a client that stopped listening out of the test output."""


@pytest.fixture
def events_server():
    """A MemMachine server for one test, on a port of its own."""
    server = EventsServer()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def home_directory(tmp_path, monkeypatch):
    """Keep the state file and Codex's home inside a temporary directory."""
    directory = tmp_path / "home"
    directory.mkdir()
    monkeypatch.setenv("HOME", str(directory))
    monkeypatch.setenv("USERPROFILE", str(directory))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return directory


def claude_transcript(path, count=1, first=1):
    """A Claude Code transcript of `count` user messages."""
    entries = [
        {
            "type": "user",
            "uuid": f"aaaaaaaa-0000-4000-8000-{index:012d}",
            "sessionId": SESSION,
            "timestamp": "2026-09-17T10:00:00.000Z",
            "message": {"role": "user", "content": f"message {index}"},
        }
        for index in range(first, first + count)
    ]
    with path.open("a", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(f"{json.dumps(entry)}\n")
    return path


def run_capture(server, transcript, monkeypatch, agent="claude-code", budget=None):
    """Run the capture the way a `Stop` hook does, over the hook's input."""
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": SESSION,
                    "transcript_path": str(transcript),
                    "cwd": str(transcript.parent),
                    "hook_event_name": "Stop",
                }
            )
        ),
    )
    arguments = [
        "agent",
        "capture",
        agent,
        "--server",
        server.url,
        "--tenant",
        TENANT,
    ]
    if budget is not None:
        arguments += ["--budget", str(budget)]
    return cli.main(arguments)


def state_of(home_directory, agent="claude-code"):
    """The marks an agent's state file holds."""
    directory = ".claude" if agent == "claude-code" else ".codex"
    path = home_directory / directory / "memmachine" / "capture-state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())["sessions"]


def test_the_new_events_are_posted_and_the_mark_moves(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=2)

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 0
    assert events_server.paths == [f"/v1/tenants/{TENANT}/events"]
    assert [event["blocks"][0]["text"] for event in events_server.batches[0]] == [
        "message 1",
        "message 2",
    ]
    assert state_of(home_directory)[SESSION] == {
        "offset": transcript.stat().st_size,
        "index": 2,
    }


def test_the_second_run_posts_only_what_is_new(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    run_capture(events_server, transcript, monkeypatch)

    assert run_capture(events_server, transcript, monkeypatch) == 0
    assert len(events_server.batches) == 1

    claude_transcript(transcript, count=1, first=2)
    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert [event["blocks"][0]["text"] for event in events_server.batches[1]] == [
        "message 2"
    ]
    assert state_of(home_directory)[SESSION]["index"] == 2


def test_events_the_server_already_holds_move_the_mark_too(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    held = (409, {"error": {"code": "event_exists", "message": "already held"}})
    events_server.answers = [held, held]

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 0
    # The batch is held, and so is the one event it holds.
    assert [len(batch) for batch in events_server.batches] == [1, 1]
    assert state_of(home_directory)[SESSION]["index"] == 1


def test_a_server_failure_leaves_the_mark_where_it_was(
    events_server, tmp_path, monkeypatch, home_directory, capsys
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    events_server.answers = [
        (500, {"error": {"code": "internal", "message": "Internal server error"}})
    ]

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 1
    assert state_of(home_directory) == {}
    assert "answered 500" in capsys.readouterr().err


def test_a_batch_the_server_already_holds_is_posted_event_by_event(
    events_server, tmp_path, monkeypatch, home_directory, capsys
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=2)
    held = (409, {"error": {"code": "event_exists", "message": "already held"}})
    # The batch carries one event the server holds and one it does not,
    # which is what a session forked from another looks like.
    events_server.answers = [held, held, (200, {"stored": []})]

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 0
    assert [len(batch) for batch in events_server.batches] == [2, 1, 1]
    assert [
        event["blocks"][0]["text"]
        for batch in events_server.batches[1:]
        for event in batch
    ] == [
        "message 1",
        "message 2",
    ]
    assert state_of(home_directory)[SESSION]["index"] == 2
    # Only the event the server did not hold was stored now.
    assert "captured 1 event of" in capsys.readouterr().err


def test_a_failure_during_the_event_by_event_pass_keeps_what_was_held(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=3)
    events_server.answers = [
        (409, {"error": {"code": "event_exists", "message": "already held"}}),
        (200, {"stored": []}),
        (500, {"error": {"code": "internal", "message": "no"}}),
    ]

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 1
    assert [len(batch) for batch in events_server.batches] == [3, 1, 1]
    # Events go in the order the transcript holds them, so the first is
    # held whatever happened after it: the mark stands past that entry
    # and the next `Stop` posts the two that are left.
    assert state_of(home_directory)[SESSION]["index"] == 1

    events_server.answers = []
    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert [len(batch) for batch in events_server.batches] == [3, 1, 1, 2]
    assert state_of(home_directory)[SESSION]["index"] == 3


def test_an_entry_is_passed_only_when_every_event_of_it_is_held(
    events_server, tmp_path, monkeypatch, home_directory
):
    # One entry of two events: a message and the tool call after it.
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "uuid": "aaaaaaaa-0000-4000-8000-000000000001",
                "sessionId": SESSION,
                "timestamp": "2026-09-17T10:00:00.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "running it"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events_server.answers = [
        (409, {"error": {"code": "event_exists", "message": "already held"}}),
        (200, {"stored": []}),
        (500, {"error": {"code": "internal", "message": "no"}}),
    ]

    assert run_capture(events_server, transcript, monkeypatch) == 1

    # The entry's first event is held and its second is not, so the mark
    # stays before the entry: an entry is read again whole.
    assert state_of(home_directory) == {}


def test_a_conflict_of_another_kind_is_a_failure(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    events_server.answers = [
        (409, {"error": {"code": "invalid_request", "message": "no"}})
    ]

    assert run_capture(events_server, transcript, monkeypatch) == 1
    assert state_of(home_directory) == {}


def test_a_server_that_does_not_answer_in_time_leaves_the_mark(
    events_server, tmp_path, monkeypatch, home_directory, capsys
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    events_server.delay = 0.5

    exit_code = run_capture(events_server, transcript, monkeypatch, budget=0.1)

    assert exit_code == 1
    assert state_of(home_directory) == {}
    assert "could not be reached" in capsys.readouterr().err


def test_a_batch_holds_at_most_two_hundred_events(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=250)

    exit_code = run_capture(events_server, transcript, monkeypatch)

    assert exit_code == 0
    assert [len(batch) for batch in events_server.batches] == [200, 50]
    assert state_of(home_directory)[SESSION]["index"] == 250


def test_a_batch_that_failed_leaves_the_one_before_it_posted(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=250)
    events_server.answers = [
        (200, {"stored": []}),
        (500, {"error": {"code": "internal", "message": "no"}}),
    ]

    assert run_capture(events_server, transcript, monkeypatch) == 1
    assert state_of(home_directory)[SESSION]["index"] == 200

    events_server.answers = []
    assert run_capture(events_server, transcript, monkeypatch) == 0

    # The first batch is not posted again, only the rest.
    assert [len(batch) for batch in events_server.batches] == [200, 50, 50]


def test_codex_keeps_its_marks_under_its_own_home(
    events_server, tmp_path, monkeypatch, home_directory
):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-17T10:00:00.000+00:00",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = run_capture(events_server, rollout, monkeypatch, agent="codex")

    assert exit_code == 0
    assert events_server.batches[0][0]["source_id"] == "codex"
    assert events_server.batches[0][0]["properties"]["agent"] == "codex"
    assert state_of(home_directory, agent="codex")[SESSION]["index"] == 1


def test_a_codex_home_of_its_own_holds_the_marks(
    events_server, tmp_path, monkeypatch, home_directory
):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("", encoding="utf-8")

    assert run_capture(events_server, rollout, monkeypatch, agent="codex") == 0
    assert not (codex_home / "memmachine").exists()

    rollout.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-17T10:00:00.000+00:00",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_capture(events_server, rollout, monkeypatch, agent="codex") == 0
    assert (codex_home / "memmachine" / "capture-state.json").exists()


def test_the_project_is_the_repository_the_session_runs_in(
    events_server, tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    working_directory = repository / "packages" / "client"
    working_directory.mkdir(parents=True)
    transcript = claude_transcript(working_directory / "session.jsonl", count=1)

    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert events_server.batches[0][0]["properties"]["project"] == str(repository)


def test_a_directory_that_is_no_checkout_is_the_project_itself(
    events_server, tmp_path, monkeypatch
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)

    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert events_server.batches[0][0]["properties"]["project"] == str(tmp_path)


def test_a_transcript_that_was_replaced_is_read_from_its_start(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=3)
    run_capture(events_server, transcript, monkeypatch)

    transcript.write_text("", encoding="utf-8")
    claude_transcript(transcript, count=1, first=9)
    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert [event["blocks"][0]["text"] for event in events_server.batches[1]] == [
        "message 9"
    ]
    assert state_of(home_directory)[SESSION]["index"] == 1


def test_the_marks_of_other_sessions_are_kept(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    state_path = home_directory / ".claude" / "memmachine" / "capture-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"sessions": {"another": {"offset": 12, "index": 3}}}),
        encoding="utf-8",
    )

    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert state_of(home_directory)["another"] == {"offset": 12, "index": 3}
    assert state_of(home_directory)[SESSION]["index"] == 1


def test_only_the_most_recent_sessions_keep_their_marks(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    state_path = home_directory / ".claude" / "memmachine" / "capture-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "sessions": {
                    f"session-{index}": {"offset": index, "index": index}
                    for index in range(coding_agent_capture._REMEMBERED_SESSIONS)
                }
            }
        ),
        encoding="utf-8",
    )

    assert run_capture(events_server, transcript, monkeypatch) == 0

    marks = state_of(home_directory)
    assert len(marks) == coding_agent_capture._REMEMBERED_SESSIONS
    assert "session-0" not in marks
    assert SESSION in marks


def test_a_state_file_that_cannot_be_read_costs_one_repeated_batch(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    state_path = home_directory / ".claude" / "memmachine" / "capture-state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("half a fi", encoding="utf-8")

    assert run_capture(events_server, transcript, monkeypatch) == 0

    assert len(events_server.batches[0]) == 1
    assert state_of(home_directory)[SESSION]["index"] == 1


def test_a_transcript_named_by_hand_needs_no_hook(
    events_server, tmp_path, monkeypatch, home_directory
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    exit_code = cli.main(
        [
            "agent",
            "capture",
            "claude-code",
            "--server",
            events_server.url,
            "--tenant",
            TENANT,
            "--transcript",
            str(transcript),
            "--session-id",
            SESSION,
        ]
    )

    assert exit_code == 0
    assert events_server.batches[0][0]["session_id"] == SESSION


def test_a_capture_with_no_transcript_to_read_says_so(
    events_server, monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"session_id": SESSION, "cwd": "/repo"}))
    )

    exit_code = cli.main(
        [
            "agent",
            "capture",
            "claude-code",
            "--server",
            events_server.url,
            "--tenant",
            TENANT,
        ]
    )

    assert exit_code == 1
    assert "no transcript to read" in capsys.readouterr().err


def test_a_transcript_that_is_not_there_is_a_failure(
    events_server, tmp_path, monkeypatch, capsys
):
    exit_code = run_capture(events_server, tmp_path / "missing.jsonl", monkeypatch)

    assert exit_code == 1
    assert "cannot be read" in capsys.readouterr().err


def test_nothing_is_written_to_standard_output(
    events_server, tmp_path, monkeypatch, capsys
):
    transcript = claude_transcript(tmp_path / "session.jsonl", count=1)

    assert run_capture(events_server, transcript, monkeypatch) == 0

    # An agent reads a hook's standard output as context, so what
    # happened is reported on standard error.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "captured 1 event of" in captured.err
