import time
from datetime import UTC, datetime

import pytest

from memmachine_server.common.neo4j_utils import coerce_datetime_to_timestamp


@pytest.mark.skipif(
    not hasattr(time, "tzset"),
    reason="tzset is POSIX-only; the process zone cannot be pinned on Windows",
)
def test_iso_string_coerces_as_utc_instant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A naive ISO string means UTC, like every other filter datetime.

    Pinned under a non-UTC process zone so a local-zone reading of the
    parsed string cannot pass by coincidence.
    """
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        expected = datetime(2024, 1, 1, 13, 30, 45, tzinfo=UTC).timestamp()
        assert coerce_datetime_to_timestamp("2024-01-01T13:30:45") == expected
        assert (
            coerce_datetime_to_timestamp(datetime(2024, 1, 1, 13, 30, 45, tzinfo=UTC))
            == expected
        )
    finally:
        # undo() restores the variable but not libc's cached zone; without
        # this tzset() the whole process stays on Los Angeles time for
        # every later test in the run.
        monkeypatch.undo()
        time.tzset()
