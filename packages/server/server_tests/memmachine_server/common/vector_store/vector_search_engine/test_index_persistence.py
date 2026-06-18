"""Tests for the shared atomic index persistence helpers."""

from pathlib import Path

import pytest

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
            assert not Path(temp).exists()
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
