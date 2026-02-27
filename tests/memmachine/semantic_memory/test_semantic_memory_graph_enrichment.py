"""Unit tests for graph-enhanced semantic retrieval in SemanticService.

Tests cover:
- RELATED_TO enrichment (8.5)
- Contradiction annotation (8.6)
- Supersession resolution (8.7)
- Graceful fallback without SemanticRelationshipStorage (8.8)
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, create_autospec

import pytest

from memmachine.common.episode_store import EpisodeStorage
from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    RawSemanticPrompt,
    SemanticCategory,
    SemanticFeature,
)
from memmachine.semantic_memory.storage.feature_relationship_types import (
    ContradictionPair,
    FeatureRelationship,
    FeatureRelationshipType,
    RelationshipDirection,
    SupersessionChain,
)
from memmachine.semantic_memory.storage.semantic_relationship_storage import (
    SemanticRelationshipStorage,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from tests.memmachine.semantic_memory.semantic_test_utils import SpyEmbedder

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)


def _feature(fid: str, value: str = "v", tag: str = "t") -> SemanticFeature:
    return SemanticFeature(
        set_id="s1",
        category="Profile",
        tag=tag,
        feature_name=f"feat_{fid}",
        value=value,
        metadata=SemanticFeature.Metadata(id=fid),
    )


def _rel(
    src: str,
    tgt: str,
    rtype: FeatureRelationshipType = FeatureRelationshipType.RELATED_TO,
    confidence: float = 0.9,
) -> FeatureRelationship:
    return FeatureRelationship(
        source_id=src,
        target_id=tgt,
        relationship_type=rtype,
        confidence=confidence,
        detected_at=_NOW,
        source="rule",
    )


# ---------------------------------------------------------------------------
# Mock storage that implements both protocols
# ---------------------------------------------------------------------------


class _MockRelStorage(SemanticStorage, SemanticRelationshipStorage):
    """Minimal mock implementing both SemanticStorage and SemanticRelationshipStorage."""

    def __init__(self) -> None:
        # Configurable return values for relationship methods.
        self._relationships: dict[str, list[FeatureRelationship]] = {}
        self._features: dict[str, SemanticFeature] = {}
        self._supersession_chains: dict[str, SupersessionChain] = {}

    # --- SemanticRelationshipStorage protocol ---

    async def add_feature_relationship(self, **kwargs) -> None:
        pass  # pragma: no cover

    async def get_feature_relationships(
        self,
        feature_id,
        *,
        relationship_type=None,
        direction=RelationshipDirection.BOTH,
        min_confidence=None,
    ) -> list[FeatureRelationship]:
        rels = self._relationships.get(feature_id, [])
        if relationship_type is not None:
            rels = [r for r in rels if r.relationship_type == relationship_type]
        return rels

    async def delete_feature_relationships(self, **kwargs) -> None:
        pass  # pragma: no cover

    async def find_contradictions(self, *, set_id) -> list[ContradictionPair]:
        return []  # pragma: no cover

    async def find_supersession_chain(self, feature_id) -> SupersessionChain:
        if feature_id in self._supersession_chains:
            return self._supersession_chains[feature_id]
        return SupersessionChain(current=feature_id, chain=[feature_id])

    # --- SemanticStorage stubs (only get_feature used by enrichment) ---

    async def get_feature(self, feature_id, load_citations=False, **kwargs):
        return self._features.get(feature_id)

    # Remaining abstract methods — not exercised.
    async def startup(self):
        pass

    async def cleanup(self):
        pass

    async def add_feature(self, **kw):
        return "id"

    async def update_feature(self, **kw):
        pass

    async def delete_features(self, fids):
        pass

    async def delete_feature_set(self, **kw):
        pass

    async def get_feature_set(self, **kw):
        return []

    async def add_history_to_set(self, **kw):
        pass

    async def delete_history(self, hids):
        pass

    async def delete_history_set(self, **kw):
        pass

    async def get_history_set_ids(self, **kw):
        return []

    async def get_history_messages_count(self, **kw):
        return 0

    async def get_set_ids_starts_with(self, prefix):
        return []

    async def reset_set_ids(self, **kw):
        pass

    async def add_citations(self, fid, cids):
        pass

    async def mark_messages_ingested(self, **kw):
        pass

    async def get_uningested_messages(self, **kw):
        return []

    async def delete_all(self):
        pass

    async def get_history_messages(self, **kw):
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rel_storage() -> _MockRelStorage:
    return _MockRelStorage()


@pytest.fixture
def service(rel_storage: _MockRelStorage) -> SemanticService:
    """Create a SemanticService wired to our mock relationship storage."""
    spy = SpyEmbedder()
    mock_llm = create_autospec(LanguageModel, instance=True)
    mock_episode = create_autospec(EpisodeStorage, instance=True)
    mock_config = create_autospec(SemanticConfigStorage, instance=True)

    prompt = RawSemanticPrompt(update_prompt="up", consolidation_prompt="con")
    cat = SemanticCategory(name="Profile", prompt=prompt)

    params = SemanticService.Params(
        semantic_storage=rel_storage,
        episode_storage=mock_episode,
        semantic_config_storage=mock_config,
        resource_manager=type(
            "_RM",
            (),
            {
                "get_embedder": AsyncMock(return_value=spy),
                "get_language_model": AsyncMock(return_value=mock_llm),
            },
        )(),
        default_embedder=spy,
        default_embedder_name="spy",
        default_language_model=mock_llm,
        default_category_retriever=lambda _: [cat],
    )
    return SemanticService(params)


# ---------------------------------------------------------------------------
# 8.5 — RELATED_TO enrichment
# ---------------------------------------------------------------------------


async def test_related_to_enrichment_adds_related_features(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """RELATED_TO relationships add new features to the result set."""
    f1 = _feature("f1", value="Alice likes coffee")
    f_related = _feature("f_related", value="Alice likes tea")

    rel_storage._relationships["f1"] = [
        _rel("f1", "f_related", FeatureRelationshipType.RELATED_TO),
    ]
    rel_storage._features["f_related"] = f_related

    result = await service._enrich_related_to([f1], rel_storage)

    ids = [f.metadata.id for f in result]
    assert "f1" in ids
    assert "f_related" in ids
    assert len(result) == 2


async def test_related_to_deduplication(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """Already-present features are not duplicated via RELATED_TO."""
    f1 = _feature("f1")
    f2 = _feature("f2")

    # f1 is related to f2, but f2 is already in results.
    rel_storage._relationships["f1"] = [
        _rel("f1", "f2", FeatureRelationshipType.RELATED_TO),
    ]

    result = await service._enrich_related_to([f1, f2], rel_storage)

    ids = [f.metadata.id for f in result]
    assert ids.count("f2") == 1
    assert len(result) == 2


async def test_related_to_no_relationships(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """When no RELATED_TO relationships exist, results are unchanged."""
    f1 = _feature("f1")

    result = await service._enrich_related_to([f1], rel_storage)

    assert len(result) == 1
    assert result[0].metadata.id == "f1"


async def test_related_to_handles_missing_feature(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """When a related feature can't be loaded, it's silently skipped."""
    f1 = _feature("f1")

    rel_storage._relationships["f1"] = [
        _rel("f1", "f_missing", FeatureRelationshipType.RELATED_TO),
    ]
    # f_missing is NOT in rel_storage._features → get_feature returns None

    result = await service._enrich_related_to([f1], rel_storage)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# 8.6 — Contradiction annotation
# ---------------------------------------------------------------------------


async def test_contradiction_annotation_sets_field(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """CONTRADICTS relationships are annotated on both features."""
    f1 = _feature("f1", value="Alice likes coffee")
    f2 = _feature("f2", value="Alice hates coffee")

    rel_storage._relationships["f1"] = [
        _rel("f1", "f2", FeatureRelationshipType.CONTRADICTS),
    ]
    rel_storage._relationships["f2"] = [
        _rel("f1", "f2", FeatureRelationshipType.CONTRADICTS),
    ]

    features = [f1, f2]
    await service._annotate_contradictions(features, rel_storage)

    f1_result = next(f for f in features if f.metadata.id == "f1")
    f2_result = next(f for f in features if f.metadata.id == "f2")

    assert f1_result.contradicted_by == ["f2"]
    assert f2_result.contradicted_by == ["f1"]


async def test_contradiction_only_within_result_set(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """Contradictions with features outside the result set are ignored."""
    f1 = _feature("f1")

    rel_storage._relationships["f1"] = [
        _rel("f1", "f_outside", FeatureRelationshipType.CONTRADICTS),
    ]

    features = [f1]
    await service._annotate_contradictions(features, rel_storage)

    assert features[0].contradicted_by is None


async def test_no_contradictions_leaves_field_none(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """Without CONTRADICTS relationships, contradicted_by stays None."""
    f1 = _feature("f1")

    features = [f1]
    await service._annotate_contradictions(features, rel_storage)

    assert features[0].contradicted_by is None


# ---------------------------------------------------------------------------
# 8.7 — Supersession resolution
# ---------------------------------------------------------------------------


async def test_supersession_replaces_with_current(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """Superseded features are replaced by their current version."""
    f_old = _feature("f_old", value="Old value")
    f_current = _feature("f_current", value="Current value")

    rel_storage._supersession_chains["f_old"] = SupersessionChain(
        current="f_current", chain=["f_current", "f_old"]
    )
    rel_storage._features["f_current"] = f_current

    result = await service._resolve_supersessions([f_old], rel_storage)

    assert len(result) == 1
    assert result[0].metadata.id == "f_current"
    assert result[0].value == "Current value"


async def test_supersession_no_chain_keeps_feature(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """Features without supersession chains are kept as-is."""
    f1 = _feature("f1")

    result = await service._resolve_supersessions([f1], rel_storage)

    assert len(result) == 1
    assert result[0].metadata.id == "f1"


async def test_supersession_deduplicates_replacement(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """When two features are superseded by the same current, only one copy appears."""
    f_old1 = _feature("f_old1", value="V1")
    f_old2 = _feature("f_old2", value="V2")
    f_current = _feature("f_current", value="Current")

    rel_storage._supersession_chains["f_old1"] = SupersessionChain(
        current="f_current", chain=["f_current", "f_old1"]
    )
    rel_storage._supersession_chains["f_old2"] = SupersessionChain(
        current="f_current", chain=["f_current", "f_old2"]
    )
    rel_storage._features["f_current"] = f_current

    result = await service._resolve_supersessions([f_old1, f_old2], rel_storage)

    ids = [f.metadata.id for f in result]
    assert ids.count("f_current") == 1
    assert len(result) == 1


async def test_supersession_missing_replacement_drops_feature(
    service: SemanticService,
    rel_storage: _MockRelStorage,
) -> None:
    """When the replacement feature can't be loaded, the old one is dropped."""
    f_old = _feature("f_old")

    rel_storage._supersession_chains["f_old"] = SupersessionChain(
        current="f_gone", chain=["f_gone", "f_old"]
    )
    # f_gone is NOT in _features

    result = await service._resolve_supersessions([f_old], rel_storage)

    assert len(result) == 0


# ---------------------------------------------------------------------------
# 8.8 — Graceful fallback
# ---------------------------------------------------------------------------


async def test_enrichment_fallback_without_relationship_storage() -> None:
    """When storage doesn't implement SemanticRelationshipStorage, features unchanged."""
    spy = SpyEmbedder()
    mock_llm = create_autospec(LanguageModel, instance=True)
    mock_episode = create_autospec(EpisodeStorage, instance=True)
    mock_config = create_autospec(SemanticConfigStorage, instance=True)
    # Use a plain SemanticStorage mock (NOT SemanticRelationshipStorage).
    mock_storage = create_autospec(SemanticStorage, instance=True)

    prompt = RawSemanticPrompt(update_prompt="up", consolidation_prompt="con")
    cat = SemanticCategory(name="Profile", prompt=prompt)

    params = SemanticService.Params(
        semantic_storage=mock_storage,
        episode_storage=mock_episode,
        semantic_config_storage=mock_config,
        resource_manager=type(
            "_RM",
            (),
            {
                "get_embedder": AsyncMock(return_value=spy),
                "get_language_model": AsyncMock(return_value=mock_llm),
            },
        )(),
        default_embedder=spy,
        default_embedder_name="spy",
        default_language_model=mock_llm,
        default_category_retriever=lambda _: [cat],
    )
    svc = SemanticService(params)

    f1 = _feature("f1")
    f2 = _feature("f2")

    result = await svc._enrich_with_relationships([f1, f2])

    assert len(result) == 2
    assert result[0].metadata.id == "f1"
    assert result[1].metadata.id == "f2"


async def test_enrichment_empty_features(
    service: SemanticService,
) -> None:
    """Empty feature list is returned unchanged."""
    result = await service._enrich_with_relationships([])
    assert result == []


async def test_contradicted_by_field_default_none() -> None:
    """New SemanticFeature has contradicted_by=None by default."""
    f = SemanticFeature(category="c", tag="t", feature_name="fn", value="v")
    assert f.contradicted_by is None


async def test_contradicted_by_field_serializable() -> None:
    """contradicted_by field can be set and serialized."""
    f = SemanticFeature(
        category="c",
        tag="t",
        feature_name="fn",
        value="v",
        contradicted_by=["id1", "id2"],
    )
    assert f.contradicted_by == ["id1", "id2"]
    d = f.model_dump()
    assert d["contradicted_by"] == ["id1", "id2"]
