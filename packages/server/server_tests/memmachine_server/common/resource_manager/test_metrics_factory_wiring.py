"""Every OperationTracker must be handed a metrics factory.

`OperationTracker` accepts ``metrics_factory=None`` and then silently discards
every timing it takes — no error, no warning, no series. A component that is
fully instrumented but never given a factory therefore looks identical to one
that was never instrumented at all, and the only way to notice is to go looking
for a metric that should exist.

That is not hypothetical: the Neo4j store, the episode store and the session
store each shipped instrumented and unwired, which is why database latency
appeared to be unmeasurable. These tests pin the wiring for the components that
cannot be reached from a default deployment — the event backend's segment store
and event memory — where a regression would otherwise go unseen until somebody
configured that backend and wondered where the numbers were.
"""

from unittest.mock import MagicMock

import pytest

from memmachine_server.common.metrics_factory import MetricsFactory


@pytest.fixture
def mock_metrics_factory():
    """A factory that records nothing but is distinguishable from None."""

    class MockMetricsFactory(MetricsFactory):
        def __init__(self):
            self.counters = MagicMock()
            self.gauge = MagicMock()
            self.histogram = MagicMock()
            self.summaries = MagicMock()

        def get_counter(self, name, description, label_names=...):
            return self.counters

        def get_summary(self, name, description, label_names=...):
            return self.summaries

        def get_gauge(self, name, description, label_names=...):
            return self.gauge

        def get_histogram(self, name, description, label_names=...):
            return self.histogram

        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    return MockMetricsFactory()


def test_get_segment_store_supplies_a_factory(monkeypatch, mock_metrics_factory):
    """The resource manager must hand the segment store a factory.

    Constructed directly, the store honours whatever it is given — so a test
    that passes a factory in proves nothing. The defect was here, at the call
    site, where none was passed at all.
    """
    import asyncio

    from memmachine_server.common.resource_manager import resource_manager as rm

    captured = {}

    class CapturingStore:
        def __init__(self, params):
            captured["params"] = params

        async def startup(self):
            return None

    monkeypatch.setattr(rm, "SQLAlchemySegmentStore", CapturingStore)
    monkeypatch.setattr(
        rm.ResourceManagerImpl,
        "get_metrics_factory",
        staticmethod(lambda name: _resolved(mock_metrics_factory)),
    )

    manager = rm.ResourceManagerImpl.__new__(rm.ResourceManagerImpl)
    manager._segment_stores = {}
    manager._segment_store_lock = asyncio.Lock()

    from sqlalchemy.ext.asyncio import AsyncEngine

    async def fake_engine(_name):
        return MagicMock(spec=AsyncEngine)

    manager.get_sql_engine = fake_engine

    asyncio.run(manager.get_segment_store("profile_storage"))

    assert captured["params"].metrics_factory is not None, (
        "get_segment_store built the store without a metrics factory, so its "
        "OperationTracker discards every timing silently"
    )


async def _resolved(value):
    """Await-able wrapper, so a plain value can stand in for a coroutine."""
    return value


def test_segment_store_params_accepts_a_factory():
    """The params model must carry the field at all.

    Guards the shape rather than the value: if the field is dropped, callers
    that pass it start failing loudly instead of wiring nothing.
    """
    from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
        SQLAlchemySegmentStoreParams,
    )

    assert "metrics_factory" in SQLAlchemySegmentStoreParams.model_fields


def test_event_backend_params_carries_a_factory_through(mock_metrics_factory):
    """EventBackendParams must be able to hand EventMemory a factory.

    EventMemory reads ``params.metrics_factory``; if the backend params have no
    such field, the value silently defaults to None however carefully the
    service locator resolves one.
    """
    from memmachine_server.episodic_memory.long_term_memory.long_term_memory import (
        EventBackendParams,
    )

    assert "metrics_factory" in EventBackendParams.model_fields

    field = EventBackendParams.model_fields["metrics_factory"]
    assert field.default is None, (
        "the field should default to None so existing callers keep working; "
        "the service locator is what supplies a real factory"
    )


def test_event_params_supplies_a_factory(mock_metrics_factory):
    """The service locator must put a factory into EventBackendParams.

    EventMemory reads ``params.metrics_factory``; if the locator never sets it,
    the field's None default silently wins and the tracker goes dark.
    """
    import inspect

    from memmachine_server.episodic_memory.long_term_memory import service_locator

    source = inspect.getsource(service_locator._event_params)
    assert "metrics_factory=" in source, (
        "_event_params builds EventBackendParams without a metrics_factory, so "
        "EventMemory's OperationTracker discards every timing silently"
    )


def test_tracker_without_a_factory_is_silent():
    """The failure mode itself, stated as a test.

    This is what the three shipped-but-unwired components were doing: taking
    timings correctly and throwing them away. It passes today and should keep
    passing — it documents why the assertions above are worth having.
    """
    from memmachine_server.common.metrics_factory import OperationTracker

    tracker = OperationTracker(None, prefix="example")

    assert tracker._histogram is None
    # No exception, no warning: the observation simply goes nowhere.
    tracker.emit("some_operation", 0.5)
