"""Tests for the default resolution of a v1 tenant: a namespaced session."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine_server.common.errors import (
    ConfigurationError,
    SessionAlreadyExistsError,
    SessionDeletedError,
    SessionNotFoundError,
)
from memmachine_server.episodic_memory.long_term_memory.service_locator import (
    partition_key_for_session,
)
from memmachine_server.server.api_v1.episodic_session_tenants import (
    TENANT_SESSION_KEY_PREFIX,
    EpisodicSessionTenantEventMemories,
    session_key_for_tenant,
)
from memmachine_server.server.api_v1.tenant_event_memories import (
    ComponentNotEnabledError,
    TenantNotFoundError,
)

_async = pytest.mark.asyncio

_TENANT = "alice"
_SESSION_KEY = "v1/tenants/alice"


@asynccontextmanager
async def _yielding(value):
    yield value


def _long_term_memory(*, event_memory=None, reranker=None):
    long_term_memory = MagicMock()
    long_term_memory.event_memory = event_memory
    long_term_memory.reranker = reranker
    return long_term_memory


def _episodic_memory(long_term_memory):
    episodic_memory = MagicMock()
    episodic_memory.long_term_memory = long_term_memory
    return episodic_memory


def _memmachine(*, opens=None, session=None):
    """An AsyncMock MemMachine whose manager opens `opens`, or raises it."""
    manager = MagicMock()
    if isinstance(opens, Exception):
        manager.open_episodic_memory = MagicMock(side_effect=opens)
    else:
        manager.open_episodic_memory = MagicMock(
            side_effect=lambda session_key: _yielding(opens)
        )
    manager.delete_episodic_session = AsyncMock()
    resources = MagicMock()
    resources.get_episodic_memory_manager = AsyncMock(return_value=manager)
    resources.get_session_data_manager = AsyncMock(return_value=AsyncMock())
    memmachine = AsyncMock()
    memmachine.resource_manager = resources
    memmachine.get_session = AsyncMock(return_value=session)
    return memmachine


async def _episodic_memory_manager(memmachine):
    return await memmachine.resource_manager.get_episodic_memory_manager()


# ===================================================================
# the namespace
# ===================================================================


class TestNamespace:
    @pytest.mark.parametrize(
        "tenant",
        ["alice", "default", "org/project", "v1", "tenants", "Ünïcødé", "a" * 255],
    )
    def test_a_tenant_key_can_never_be_a_v2_session_key(self, tenant):
        session_key = session_key_for_tenant(tenant)
        assert session_key.startswith(TENANT_SESSION_KEY_PREFIX)
        organization, _, project = session_key.partition("/")
        assert organization == "v1"
        # A v2 session key is an organization id and a project id joined by
        # a slash, and neither id may hold one.
        assert "/" in project

    def test_the_partition_differs_from_the_v2_project_of_the_same_name(self):
        assert partition_key_for_session(
            session_key_for_tenant("alice")
        ) != partition_key_for_session("alice/alice")


# ===================================================================
# lifecycle
# ===================================================================


class TestLifecycle:
    @_async
    async def test_create_asks_for_the_event_backend_and_no_short_term_memory(self):
        memmachine = _memmachine(
            opens=_episodic_memory(_long_term_memory(event_memory=object()))
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        assert await tenants.create(_TENANT) is True

        memmachine.create_session.assert_awaited_once()
        arguments = memmachine.create_session.await_args
        assert arguments.args[0] == _SESSION_KEY
        configuration = arguments.kwargs["user_conf"]
        assert configuration.long_term_memory.backend == "event"
        assert configuration.long_term_memory_enabled is True
        assert configuration.short_term_memory_enabled is False

    @_async
    async def test_create_answers_false_for_a_tenant_that_exists(self):
        memmachine = _memmachine(
            opens=_episodic_memory(_long_term_memory(event_memory=object()))
        )
        memmachine.create_session = AsyncMock(
            side_effect=SessionAlreadyExistsError(_SESSION_KEY)
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        assert await tenants.create(_TENANT) is False

    @_async
    async def test_create_reports_a_configuration_failure_as_the_component(self):
        memmachine = _memmachine()
        memmachine.create_session = AsyncMock(
            side_effect=ConfigurationError("no vector store")
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        with pytest.raises(ComponentNotEnabledError, match="no vector store"):
            await tenants.create(_TENANT)

    @_async
    async def test_exists_follows_the_session(self):
        memmachine = _memmachine(session=None)
        tenants = EpisodicSessionTenantEventMemories(memmachine)
        assert await tenants.exists(_TENANT) is False

        memmachine.get_session = AsyncMock(return_value=MagicMock())
        assert await tenants.exists(_TENANT) is True
        memmachine.get_session.assert_awaited_with(_SESSION_KEY)

    @_async
    async def test_delete_drops_the_memory_and_the_session_row(self):
        memmachine = _memmachine()
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        await tenants.delete(_TENANT)

        manager = await _episodic_memory_manager(memmachine)
        manager.delete_episodic_session.assert_awaited_once_with(_SESSION_KEY)
        session_data_manager = (
            await memmachine.resource_manager.get_session_data_manager()
        )
        session_data_manager.delete_session.assert_awaited_once_with(_SESSION_KEY)

    @_async
    async def test_delete_of_an_unknown_tenant_leaves_the_row_alone(self):
        memmachine = _memmachine()
        manager = await _episodic_memory_manager(memmachine)
        manager.delete_episodic_session = AsyncMock(
            side_effect=SessionNotFoundError(_SESSION_KEY)
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        await tenants.delete(_TENANT)

        session_data_manager = (
            await memmachine.resource_manager.get_session_data_manager()
        )
        session_data_manager.delete_session.assert_not_awaited()


# ===================================================================
# resolution
# ===================================================================


class TestResolution:
    @_async
    async def test_resolve_gives_the_event_memory_its_reranker_and_defaults(self):
        event_memory = object()
        reranker = object()
        memmachine = _memmachine(
            opens=_episodic_memory(
                _long_term_memory(event_memory=event_memory, reranker=reranker)
            )
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        resolved = await tenants.resolve(_TENANT)

        assert resolved.event_memory is event_memory
        assert resolved.reranker is reranker
        assert resolved.defaults.query_limit > 0
        manager = await _episodic_memory_manager(memmachine)
        manager.open_episodic_memory.assert_called_once_with(_SESSION_KEY)

    @pytest.mark.parametrize(
        "failure",
        [SessionNotFoundError(_SESSION_KEY), SessionDeletedError(_SESSION_KEY)],
    )
    @_async
    async def test_a_session_that_is_not_there_is_an_unknown_tenant(self, failure):
        tenants = EpisodicSessionTenantEventMemories(_memmachine(opens=failure))

        with pytest.raises(TenantNotFoundError, match=_TENANT):
            await tenants.resolve(_TENANT)

    @_async
    async def test_a_session_without_long_term_memory_has_no_component(self):
        memmachine = _memmachine(opens=_episodic_memory(None))
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        with pytest.raises(ComponentNotEnabledError, match="not enabled"):
            await tenants.resolve(_TENANT)

    @_async
    async def test_a_declarative_session_has_no_event_memory(self):
        memmachine = _memmachine(
            opens=_episodic_memory(_long_term_memory(event_memory=None))
        )
        tenants = EpisodicSessionTenantEventMemories(memmachine)

        with pytest.raises(ComponentNotEnabledError, match="event memory"):
            await tenants.resolve(_TENANT)

    @pytest.mark.parametrize(
        "failure",
        [ConfigurationError("no embedder"), ValueError("No memory is configured")],
    )
    @_async
    async def test_a_memory_that_cannot_be_built_has_no_component(self, failure):
        tenants = EpisodicSessionTenantEventMemories(_memmachine(opens=failure))

        with pytest.raises(ComponentNotEnabledError):
            await tenants.resolve(_TENANT)
