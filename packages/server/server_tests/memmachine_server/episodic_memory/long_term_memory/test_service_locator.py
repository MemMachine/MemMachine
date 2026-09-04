"""Unit tests for service_locator helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast, override

import pytest

if TYPE_CHECKING:
    # `httpx` is the optional `duckling` extra; only the duckling test needs it.
    import httpx

from memmachine_server.common.configuration.episodic_config import (
    DateparserTemporalExtractorConf,
    DucklingTemporalExtractorConf,
    ExtractorTemporalQueryPlannerConf,
    LanguageModelTemporalExtractorConf,
    LanguageModelTemporalQueryPlannerConf,
    PassthroughSegmenterConf,
    TemporalSegmenterConf,
)
from memmachine_server.common.language_model.language_model import LanguageModel
from memmachine_server.common.resource_manager import CommonResourceManager
from memmachine_server.episodic_memory.event_memory.segmenter.temporal_segmenter import (
    TemporalSegmenter,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    _build_segmenter,
    _build_temporal_extractor,
    _build_temporal_query_planner,
    _resolve_user_properties_schema,
    partition_key_for_session,
)
from memmachine_server.temporal.extractor.duckling_temporal_extractor import (
    DucklingTemporalExtractor,
)
from memmachine_server.temporal.extractor.extractor_temporal_query_planner import (
    ExtractorTemporalQueryPlanner,
)
from memmachine_server.temporal.extractor.language_model_temporal_extractor import (
    LanguageModelTemporalExtractor,
)
from memmachine_server.temporal.query_planner.language_model_temporal_query_planner import (
    LanguageModelTemporalQueryPlanner,
)

_PARTITION_KEY_RE = re.compile(r"^[a-z0-9_]+$")
_PARTITION_KEY_MAX_LEN = 32


def _is_valid_partition_key(value: str) -> bool:
    return bool(_PARTITION_KEY_RE.match(value)) and len(value) <= _PARTITION_KEY_MAX_LEN


def test_partition_key_passes_through_when_already_valid():
    assert partition_key_for_session("abc_123") == "abc_123"
    assert partition_key_for_session("session_42") == "session_42"


def test_partition_key_hashes_when_session_id_invalid():
    # Hyphens, uppercase, and other non-`[a-z0-9_]` chars trigger hashing.
    key = partition_key_for_session("Session-Mixed-Case-123")
    assert key != "Session-Mixed-Case-123"
    assert _is_valid_partition_key(key)
    assert len(key) == _PARTITION_KEY_MAX_LEN


def test_partition_key_hashes_when_too_long():
    long_id = "a" * 64
    key = partition_key_for_session(long_id)
    assert _is_valid_partition_key(key)
    assert len(key) == _PARTITION_KEY_MAX_LEN
    assert key != long_id


def test_partition_key_is_deterministic():
    """Same session_id always produces the same partition_key."""
    sid = "abc-123-uuid-shaped"
    assert partition_key_for_session(sid) == partition_key_for_session(sid)


def test_partition_key_distinct_inputs_produce_distinct_outputs():
    a = partition_key_for_session("abc-123-different-input-1")
    b = partition_key_for_session("abc-123-different-input-2")
    assert a != b


def test_partition_key_handles_unicode():
    # UTF-8 multi-byte input forces hashing because non-ASCII chars don't
    # match `[a-z0-9_]`.
    key = partition_key_for_session("日本語_セッション")
    assert _is_valid_partition_key(key)


def test_partition_key_empty_string_passthrough():
    """Empty session_id has length 0 but does not match `[a-z0-9_]+` (requires +)."""
    # The regex `^[a-z0-9_]+$` requires at least one char, so empty string
    # should be hashed (deterministic 32-hex digest).
    key = partition_key_for_session("")
    assert _is_valid_partition_key(key)
    assert len(key) == _PARTITION_KEY_MAX_LEN


def test_resolve_user_properties_schema_accepts_normal_keys():
    resolved = _resolve_user_properties_schema({"customer_tier": "str", "score": "int"})
    assert resolved == {"customer_tier": str, "score": int}


def test_resolve_user_properties_schema_rejects_underscore_prefixed_keys():
    """`_`-prefixed keys collide with system-defined event fields
    (`_episode_uid`, `_session_key`, ...). The merged collection schema is
    a dict-spread with user_schema last, so allowing them would silently
    overwrite the system slot and may change its declared type."""
    with pytest.raises(ValueError, match="reserved"):
        _resolve_user_properties_schema({"_episode_uid": "str"})

    with pytest.raises(ValueError, match="reserved"):
        _resolve_user_properties_schema({"_my_field": "int"})


def test_resolve_user_properties_schema_rejects_unknown_type_name():
    with pytest.raises(ValueError, match="unknown type name"):
        _resolve_user_properties_schema({"customer_tier": "date"})


# --- temporal segmenter / extractor construction ---------------------------


class _StubLanguageModel(LanguageModel):
    """A LanguageModel that satisfies InstanceOf but is never invoked here."""

    @override
    async def generate_parsed_response(self, *args, **kwargs):
        raise NotImplementedError

    @override
    async def generate_response(self, *args, **kwargs):
        raise NotImplementedError

    @override
    async def generate_response_with_token_usage(self, *args, **kwargs):
        raise NotImplementedError


class _StubResourceManager:
    """Minimal resource manager exposing only the methods used here."""

    def __init__(self, language_model: LanguageModel | None) -> None:
        self._language_model = language_model
        self.requested: list[str] = []
        self._http_client: httpx.AsyncClient | None = None

    async def get_language_model(
        self, name: str, validate: bool = False
    ) -> LanguageModel:
        self.requested.append(name)
        assert self._language_model is not None
        return self._language_model

    async def get_http_client(self) -> httpx.AsyncClient:
        import httpx

        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()


def _resource_manager(
    language_model: LanguageModel | None = None,
) -> CommonResourceManager:
    return cast(CommonResourceManager, _StubResourceManager(language_model))


@pytest.mark.asyncio
async def test_build_temporal_extractor_dateparser():
    pytest.importorskip("dateparser")
    from memmachine_server.temporal.extractor.dateparser_temporal_extractor import (
        DateparserTemporalExtractor,
    )

    extractor = await _build_temporal_extractor(
        DateparserTemporalExtractorConf(), _resource_manager()
    )
    assert isinstance(extractor, DateparserTemporalExtractor)


@pytest.mark.asyncio
async def test_build_temporal_extractor_language_model():
    manager = _StubResourceManager(_StubLanguageModel())
    extractor = await _build_temporal_extractor(
        LanguageModelTemporalExtractorConf(language_model="my-lm"),
        cast(CommonResourceManager, manager),
    )
    assert isinstance(extractor, LanguageModelTemporalExtractor)
    assert manager.requested == ["my-lm"]


@pytest.mark.asyncio
async def test_build_temporal_extractor_duckling():
    pytest.importorskip("httpx")
    manager = _StubResourceManager(None)
    extractor = await _build_temporal_extractor(
        DucklingTemporalExtractorConf(url="http://duck.test/parse"),
        cast(CommonResourceManager, manager),
    )
    assert isinstance(extractor, DucklingTemporalExtractor)
    # The client is owned by the resource manager, not the extractor: it is the
    # manager's shared client, reused across extractors.
    assert extractor._client is await manager.get_http_client()
    # This stub has no close() lifecycle, so close the shared client here.
    await manager.aclose()


@pytest.mark.asyncio
async def test_build_segmenter_temporal_wraps_base():
    pytest.importorskip("dateparser")
    segmenter = await _build_segmenter(
        TemporalSegmenterConf(
            extractor=DateparserTemporalExtractorConf(),
            base_segmenter=PassthroughSegmenterConf(),
        ),
        _resource_manager(),
    )
    assert isinstance(segmenter, TemporalSegmenter)


# --- temporal query planner construction -----------------------------------


@pytest.mark.asyncio
async def test_build_temporal_query_planner_language_model():
    manager = _StubResourceManager(_StubLanguageModel())
    planner = await _build_temporal_query_planner(
        LanguageModelTemporalQueryPlannerConf(language_model="planner-lm"),
        cast(CommonResourceManager, manager),
    )
    assert isinstance(planner, LanguageModelTemporalQueryPlanner)
    assert manager.requested == ["planner-lm"]


@pytest.mark.asyncio
async def test_build_temporal_query_planner_extractor_reuses_extractor_builder():
    pytest.importorskip("dateparser")
    planner = await _build_temporal_query_planner(
        ExtractorTemporalQueryPlannerConf(extractor=DateparserTemporalExtractorConf()),
        _resource_manager(),
    )
    assert isinstance(planner, ExtractorTemporalQueryPlanner)
