"""
Atomic on-disk index persistence helpers, shared by vector search engines.

Each engine owns its index file layout, so the atomic-swap logic lives here
(shared by the engines) rather than in the vector store: the index save
location and the number of files written differ across engine implementations.

The core guarantee is that a reader never observes a partially written index at
the target path. The index is written to a sibling temp file and swapped into
place with ``Path.replace`` (``os.replace``), which is atomic on POSIX and
Windows when source and destination share a filesystem (guaranteed here, since
the temp file is a sibling of the target). A save that is interrupted (crash,
exception) therefore leaves the previous index untouched rather than corrupting
it, which matters because the vector store treats a saved-but-unloadable index
as a hard error rather than silently rebuilding it empty.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

# Deterministic suffix so a crash leaves at most one stale temp per index file.
# The next save overwrites it and load clears it, rather than accumulating
# uniquely named leftovers.
_TEMP_SUFFIX = ".tmp"


def _temp_path(path: str) -> str:
    """Return the sibling temp path used while writing the index at `path`."""
    return f"{path}{_TEMP_SUFFIX}"


def clear_stale_index_temp(path: str) -> None:
    """
    Remove a temp file left behind by a previously interrupted save.

    Call this on load/startup so a save that crashed before the atomic swap
    does not leak a temp file across restarts. A missing temp file is a no-op.

    Args:
        path (str):
            The index file path whose sibling temp file should be cleared.
    """
    Path(_temp_path(path)).unlink(missing_ok=True)


@contextlib.contextmanager
def atomic_index_write(path: str) -> Iterator[str]:
    """
    Write an index to a temp file, then atomically swap it into `path`.

    Yields a sibling temp path for the caller to write the index to. On normal
    exit the temp file is flushed and atomically renamed onto `path`, so a
    reader sees either the old index or the new one, never a partial write. If
    the body raises, the temp file is removed and the exception propagates,
    leaving any existing index at `path` intact.

    Args:
        path (str):
            The final index file path to swap the written index into.

    Yields:
        str:
            The temp path to write the index to.
    """
    temp = _temp_path(path)
    # Clear any temp left by a previously interrupted save before reusing it.
    Path(temp).unlink(missing_ok=True)
    try:
        yield temp
        _flush_to_disk(temp)
        Path(temp).replace(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _flush_to_disk(path: str) -> None:
    """
    Best-effort fsync so the temp file's bytes are durable before the swap.

    Guards against a crash where the rename is durable but the data behind it
    is not. Failures are ignored: durability is a bonus on top of the atomic
    swap, which is the actual correctness guarantee.
    """
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
