"""
Atomic on-disk index publication, shared by vector search engines.

Each engine owns its index file layout, so the atomic-swap logic lives here
(shared by the engines) rather than in the vector store: the index save
location and the number of files written differ across engine implementations.

What a save guarantees
----------------------

A reader never observes a partially written index at the target path. The index
is written to a sibling temp file and swapped into place with ``Path.replace``
(``os.replace``), which is atomic on POSIX and Windows when source and
destination share a filesystem (guaranteed here, since the temp file is a
sibling of the target). A save that is interrupted (crash, exception) therefore
leaves the previous index untouched rather than corrupting it, which matters
because the vector store treats a saved-but-unloadable index as a hard error
rather than silently rebuilding it empty.

What it does not
----------------

The publication is atomic, not durable. A rename changes a *directory entry*,
and a directory entry cannot be flushed portably: POSIX needs an fsync on a
directory file descriptor, which network and FUSE filesystems may refuse, and
Windows has no equivalent at all. SQLite's own VFS is the precedent -- it
threads a directory-sync flag through every commit-relevant directory
operation, honors it in ``unixDelete``, and declares it
``/* Not used on win32 */`` in ``winDelete``.

So a power failure can roll a save back to the previously published index even
though ``save`` returned, while the SQLite side -- including the
pending-operation trim that ran behind that save -- stays committed. What that
leaves is records whose vectors are missing from the index: they still resolve
by uuid, but they cannot be found by search until they are re-upserted.

That is the direction this store tolerates, and it is a deliberate trade. A
stronger guarantee needs a commit protocol that never uses a directory
operation as its commit point (two index slots plus a generation record, or an
equivalent), which every engine would then have to implement and maintain. The
failure it buys out is bounded -- search recall for at most the records applied
since the last checkpoint, repaired by re-ingesting them -- so the machinery
costs more than it saves.
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

    The swap is atomic, not durable: after a power failure the index at `path`
    may be the previous one. See the module docstring for what that costs.

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
    is not, which is the direction that costs: a published index that will not
    parse is a hard error, while a publication that reverts is only missing
    vectors. Failures are ignored -- this narrows a window rather than closing
    one, since the swap itself is not durable either.
    """
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
