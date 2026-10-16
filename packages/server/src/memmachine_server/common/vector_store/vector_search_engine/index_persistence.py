"""
Atomic on-disk index publication, shared by vector search engines.

The index is written to a sibling temp file, flushed, and swapped into place
with ``os.replace``. Because the flush completes before the rename is issued,
a crash leaves either the previous index or the new one -- never the new name
over incomplete bytes, which is the case that matters: the vector store treats
a saved-but-unloadable index as a hard error, while an index that reverts is
only missing vectors until they are upserted again.

The rename itself is not made durable -- that needs an fsync on a directory
fd, which POSIX offers unevenly and Windows not at all -- so a power failure
can still roll a save back to the previously published index after ``save``
returned, while the SQLite side stays committed. Records applied since the
last checkpoint then resolve by uuid but cannot be found by search. That is
the direction this store tolerates; #1588 records why.

Why a rename at all: the stronger answer is to put the commit point inside the
file, where an fsync reaches it portably. SQLite never renames -- it commits by
truncating or zeroing its rollback journal, and in WAL mode by appending frames
whose checksums make a torn tail self-identifying, so recovery keeps everything
up to the last frame that verifies. Both need the writer to own the file
format. A search engine owns its own and exposes ``save(path)``, so above that
call a rename is the only atomicity primitive left. An engine whose format
already commits that way needs none of this.
"""

import contextlib
import os
import sys
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

    The swap is atomic, not durable: after a power failure the index at `path`
    may be the previous one.

    The body must write the yielded path **in place** -- opening it, or
    truncating and rewriting it, is fine; replacing it is not. A descriptor on
    the temp file is held open across the body so that the flush reports errors
    from the body's own write (see `_flush_to_disk`), and a body that builds a
    different file and renames it over the yielded path leaves that descriptor
    on an orphaned inode. That is checked rather than trusted: it raises
    `OSError` and publishes nothing, so a caller that breaks the rule finds out
    at its first save rather than at a power cut.

    Args:
        path (str):
            The final index file path to swap the written index into.

    Yields:
        str:
            The temp path to write the index to, in place.

    Raises:
        OSError:
            If the yielded path was replaced rather than written in place, or
            if the index bytes could not be flushed to disk. In both cases any
            existing index at `path` is left as it was.
    """
    temp = _temp_path(path)
    # Clear any temp left by a previously interrupted save before reusing it.
    Path(temp).unlink(missing_ok=True)
    # Opened before the caller writes and held across that write, so the fsync
    # below is on a descriptor that predates it. See `_flush_to_disk`.
    fd = os.open(temp, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            yield temp
            _flush_to_disk(fd, temp)
        finally:
            os.close(fd)
        Path(temp).replace(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _flush_to_disk(fd: int, path: str) -> None:
    """
    Fsync the index bytes, on a descriptor that predates them.

    This is what rules out publishing the new name over incomplete bytes: the
    data is durable before the rename is issued, and a durable write does not
    un-happen. A failed fsync therefore fails the save -- the caller discards
    the temp and the previously published index stands -- because a failure
    here is exactly the evidence that the bytes are not safe to publish.

    The descriptor has to predate the write for that to hold. Flushing would
    work on one opened afterwards, since dirty pages belong to the file rather
    than to the descriptor that dirtied them, but error reporting would not:
    Linux samples the writeback error sequence when a file is opened, so a
    descriptor opened after an error was recorded never learns of it and the
    fsync returns success over bytes already known bad.

    Predating the write is also what makes the in-place rule `atomic_index_write`
    states enforceable here: a body that replaced the file rather than writing
    it leaves this descriptor on an orphaned inode, and fsyncing that would
    report on a file nobody is about to publish.
    """
    written = Path(path).stat()
    held = os.fstat(fd)
    if (held.st_dev, held.st_ino) != (written.st_dev, written.st_ino):
        raise OSError(
            f"{path} was replaced while it was being written, so the "
            f"descriptor held across the write no longer refers to it"
        )
    _fsync(fd)


def _fsync(fd: int) -> None:
    """
    Flush `fd` as hard as the platform can be asked to.

    Darwin's `fsync` returns once the data reaches the drive, which may hold it
    in a volatile write cache. That is not enough here. The data and the rename
    that follows it land in the same cache with nothing ordering them, and the
    rename is a few bytes of metadata against an index of megabytes, so a drive
    flushing as it pleases can easily put the new name on media while the bytes
    behind it are still queued -- the torn publication this module exists to
    prevent. `F_FULLFSYNC` forces the cache out, which restores the ordering.

    A failure is not softened into `fsync`. Falling back would answer a request
    for ordering with a flush that does not provide it under power loss, and
    say nothing -- and a filesystem that cannot order data ahead of a rename is
    not one to publish an index onto. Point `index_directory` at storage that
    can. This also means no errno is inspected: refusal and failure take the
    same path, so there is no line to draw and no list to get wrong.

    Elsewhere `os.fsync` already reaches the device -- `fsync` on Linux,
    `FlushFileBuffers` on Windows.
    """
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        return

    os.fsync(fd)
