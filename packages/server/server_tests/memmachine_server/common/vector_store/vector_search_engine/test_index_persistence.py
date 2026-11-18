"""Tests for the shared atomic index persistence helpers."""

from pathlib import Path

import pytest

from memmachine_server.common.vector_store.vector_search_engine import (
    index_persistence,
)
from memmachine_server.common.vector_store.vector_search_engine.index_persistence import (
    atomic_index_write,
    clear_stale_index_temp,
)


class TestAtomicIndexWrite:
    def test_swaps_temp_into_place_on_success(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("OLD")

        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")
            # Until the context exits, the target still holds the old contents.
            assert path.read_text() == "OLD"

        assert path.read_text() == "NEW"
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_creates_target_when_missing(self, tmp_path: Path):
        path = tmp_path / "index.idx"

        with atomic_index_write(str(path)) as temp:
            Path(temp).write_text("NEW")

        assert path.read_text() == "NEW"

    def test_preserves_existing_index_on_failure(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("GOOD")

        def write_then_fail() -> None:
            with atomic_index_write(str(path)) as temp:
                Path(temp).write_text("PARTIAL")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            write_then_fail()

        # The existing index is untouched and the temp file is cleaned up.
        assert path.read_text() == "GOOD"
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_does_not_create_target_on_failure(self, tmp_path: Path):
        path = tmp_path / "index.idx"

        def write_then_fail() -> None:
            with atomic_index_write(str(path)) as temp:
                Path(temp).write_text("PARTIAL")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            write_then_fail()

        assert not path.exists()
        assert not (tmp_path / "index.idx.tmp").exists()

    def test_clears_stale_temp_before_writing(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        # A temp file left by a previously interrupted save.
        (tmp_path / "index.idx.tmp").write_text("STALE")

        with atomic_index_write(str(path)) as temp:
            # The temp is opened for the caller to write into, so what matters
            # is that none of the stale content survived into it.
            assert Path(temp).read_text() == ""
            Path(temp).write_text("NEW")

        assert path.read_text() == "NEW"


class TestClearStaleIndexTemp:
    def test_removes_leftover_temp(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        temp = tmp_path / "index.idx.tmp"
        temp.write_text("STALE")

        clear_stale_index_temp(str(path))

        assert not temp.exists()

    def test_no_temp_is_noop(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        # Must not raise when there is nothing to clear.
        clear_stale_index_temp(str(path))

    def test_leaves_index_file_untouched(self, tmp_path: Path):
        path = tmp_path / "index.idx"
        path.write_text("GOOD")

        clear_stale_index_temp(str(path))

        assert path.read_text() == "GOOD"


class TestFlushFailsTheSave:
    def test_failed_fsync_leaves_the_previous_index(self, tmp_path: Path, monkeypatch):
        """An fsync that fails must not publish; it is evidence the bytes are bad."""
        target = tmp_path / "index.idx"
        target.write_bytes(b"old")

        def failing_fsync(fd: int) -> None:
            raise OSError(5, "Input/output error")

        # Patched at the module's own flush, which is what every platform
        # reaches; `os.fsync` is not it on Darwin.
        monkeypatch.setattr(index_persistence, "_fsync", failing_fsync)

        with (
            pytest.raises(OSError, match="Input/output error"),
            atomic_index_write(str(target)) as temp,
        ):
            Path(temp).write_bytes(b"new")

        assert target.read_bytes() == b"old"
        assert not Path(f"{target}.tmp").exists()


class TestFlushGuards:
    def test_a_writer_that_replaces_the_file_fails_the_save(self, tmp_path: Path):
        """Holding a descriptor across the write assumes the write is in place.

        A caller that builds its own file and renames it over the temp leaves
        the held descriptor on an orphaned inode, so the fsync would report on
        a file nobody is about to publish. That must fail the save, not pass it.
        """
        target = tmp_path / "index.idx"
        target.write_bytes(b"old")

        def replace_the_temp_instead_of_writing_it(temp: str) -> None:
            sneaky = tmp_path / "elsewhere"
            sneaky.write_bytes(b"new")
            sneaky.replace(temp)

        with (
            pytest.raises(OSError, match="replaced while it was being written"),
            atomic_index_write(str(target)) as temp,
        ):
            replace_the_temp_instead_of_writing_it(temp)

        assert target.read_bytes() == b"old"
        assert not Path(f"{target}.tmp").exists()
