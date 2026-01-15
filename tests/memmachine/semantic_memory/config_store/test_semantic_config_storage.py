from typing import cast

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from memmachine.semantic_memory.config_store.caching_semantic_config_storage import (
    CachingSemanticConfigStorage,
)
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.config_store.config_store_sqlalchemy import (
    BaseSemanticConfigStore,
    Category,
    SemanticConfigStorageSqlAlchemy,
    Tag,
)
from memmachine.semantic_memory.semantic_model import StructuredSemanticPrompt


@pytest.fixture(
    params=[True, False],
)
def with_cache(request):
    return request.param


@pytest_asyncio.fixture
async def semantic_config_storage(sqlalchemy_engine: AsyncEngine, with_cache: bool):
    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(BaseSemanticConfigStore.metadata.drop_all)
        await conn.run_sync(BaseSemanticConfigStore.metadata.create_all)

    storage = SemanticConfigStorageSqlAlchemy(sqlalchemy_engine)

    if with_cache:
        storage = CachingSemanticConfigStorage(storage)

    yield storage

    async with sqlalchemy_engine.begin() as conn:
        await conn.run_sync(BaseSemanticConfigStore.metadata.drop_all)


@pytest.mark.asyncio
async def test_setid_config_round_trip(
    semantic_config_storage: SemanticConfigStorage,
):
    await semantic_config_storage.set_setid_config(
        set_id="set-a",
        embedder_name="embed-a",
        llm_name="llm-a",
    )

    config = await semantic_config_storage.get_setid_config(set_id="set-a")

    assert config.embedder_name == "embed-a"
    assert config.llm_name == "llm-a"
    assert config.categories == []


@pytest.mark.asyncio
async def test_upsert_and_category_retrieval(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-b"
    await semantic_config_storage.set_setid_config(
        set_id=set_id,
        embedder_name="embed-a",
        llm_name="llm-a",
    )

    category_id = await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="interests",
        prompt="What the user cares about",
    )
    await semantic_config_storage.add_tag(
        category_id=category_id,
        tag_name="music",
        description="Preferred music genres",
    )
    await semantic_config_storage.add_tag(
        category_id=category_id,
        tag_name="food",
        description="Favorite foods",
    )

    await semantic_config_storage.set_setid_config(
        set_id=set_id,
        embedder_name="embed-b",
        llm_name="llm-b",
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert config.embedder_name == "embed-b"
    assert config.llm_name == "llm-b"
    assert len(config.categories) == 1

    category = config.categories[0]
    assert category.id == category_id
    assert category.name == "interests"
    assert (
        cast(StructuredSemanticPrompt, category.prompt).description
        == "What the user cares about"
    )
    assert cast(StructuredSemanticPrompt, category.prompt).tags == {
        "food": "Favorite foods",
        "music": "Preferred music genres",
    }


@pytest.mark.asyncio
async def test_add_and_get_category(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-c"
    category_id = await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="behavior",
        prompt="User behavior",
        description="Description of user behavior",
    )

    category = await semantic_config_storage.get_category(category_id=category_id)

    assert category is not None
    assert category.id == category_id
    assert category.name == "behavior"
    assert category.prompt == "User behavior"
    assert category.description == "Description of user behavior"


@pytest.mark.asyncio
async def test_get_non_existent_category(
    semantic_config_storage: SemanticConfigStorage,
):
    category = await semantic_config_storage.get_category(category_id="non-existent")
    assert category is None

    category = await semantic_config_storage.get_category(category_id="523")
    assert category is None


@pytest.mark.asyncio
async def test_remove_category_from_setid(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-c"
    category_id = await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="behavior",
        prompt="User behavior",
    )

    with_category = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert [c.id for c in with_category.categories] == [category_id]

    await semantic_config_storage.delete_category(category_id=category_id)

    without_category = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert without_category.categories == []

    category = await semantic_config_storage.get_category(category_id=category_id)
    assert category is None


@pytest.mark.asyncio
async def test_remove_category_deletes_tags(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-c"
    category_id = await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="behavior",
        prompt="User behavior",
    )

    category = await semantic_config_storage.get_category(category_id=category_id)
    assert category is not None

    tag_id = await semantic_config_storage.add_tag(
        category_id=category_id,
        tag_name="old-tag",
        description="Old description",
    )
    tag = await semantic_config_storage.get_tag(tag_id=tag_id)
    assert tag is not None

    await semantic_config_storage.delete_category(category_id=category_id)

    tag = await semantic_config_storage.get_tag(tag_id=tag_id)
    assert tag is None


@pytest.mark.asyncio
async def test_update_and_delete_tags(
    semantic_config_storage: SemanticConfigStorage,
    sqlalchemy_engine: AsyncEngine,
):
    set_id = "set-d"
    await semantic_config_storage.set_setid_config(set_id=set_id)

    category_id = await semantic_config_storage.create_category(
        category_name="profile",
        prompt="Profile details",
        set_id=set_id,
    )

    await semantic_config_storage.add_tag(
        category_id=category_id,
        tag_name="old-tag",
        description="Old description",
    )

    session_factory = async_sessionmaker(bind=sqlalchemy_engine, expire_on_commit=False)

    async with session_factory() as session:
        tag_id = str((await session.execute(select(Tag.id))).scalar_one())

    await semantic_config_storage.update_tag(
        tag_id=tag_id,
        tag_name="updated-tag",
        tag_description="Updated description",
    )

    updated_config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert cast(StructuredSemanticPrompt, updated_config.categories[0].prompt).tags == {
        "updated-tag": "Updated description",
    }

    await semantic_config_storage.delete_tag(tag_id=tag_id)

    config_after_delete = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert (
        cast(StructuredSemanticPrompt, config_after_delete.categories[0].prompt).tags
        == {}
    )


@pytest.mark.asyncio
async def test_clone_category_copies_tags(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-e"
    await semantic_config_storage.set_setid_config(set_id=set_id)

    original_category_id = await semantic_config_storage.create_category(
        category_name="interests",
        prompt="What the user cares about",
        set_id=set_id,
    )

    await semantic_config_storage.add_tag(
        category_id=original_category_id,
        tag_name="music",
        description="Preferred music genres",
    )
    await semantic_config_storage.add_tag(
        category_id=original_category_id,
        tag_name="food",
        description="Favorite foods",
    )

    cloned_category_id = await semantic_config_storage.clone_category(
        category_id=original_category_id,
        new_name="cloned-interests",
        new_set_id=set_id,
    )
    assert cloned_category_id != original_category_id

    config = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert len(config.categories) == 2
    assert (
        config.categories[1].prompt.update_prompt
        == config.categories[0].prompt.update_prompt
    )

    assert {c.name for c in config.categories} == {"interests", "cloned-interests"}

    assert cast(StructuredSemanticPrompt, config.categories[0].prompt).tags == {
        "music": "Preferred music genres",
        "food": "Favorite foods",
    }


@pytest.mark.asyncio
async def test_add_category_to_setid_rejects_duplicate_names(
    semantic_config_storage: SemanticConfigStorage,
):
    set_id = "set-e-duplicate"
    await semantic_config_storage.set_setid_config(set_id=set_id)

    await semantic_config_storage.create_category(
        category_name="interests",
        prompt="First interests description",
        set_id=set_id,
    )

    with pytest.raises(IntegrityError):
        await semantic_config_storage.create_category(
            category_name="interests",
            prompt="Conflicting interests description",
            set_id=set_id,
        )


@pytest.mark.asyncio
async def test_delete_category_removes_tags_and_association(
    semantic_config_storage: SemanticConfigStorage,
    sqlalchemy_engine: AsyncEngine,
):
    set_id = "set-f"
    await semantic_config_storage.set_setid_config(set_id=set_id)

    category_id = await semantic_config_storage.create_category(
        category_name="interests",
        prompt="What the user cares about",
        set_id=set_id,
    )
    await semantic_config_storage.add_tag(
        category_id=category_id,
        tag_name="music",
        description="Preferred music genres",
    )

    await semantic_config_storage.delete_category(category_id=category_id)

    config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert config.categories == []

    session_factory = async_sessionmaker(bind=sqlalchemy_engine, expire_on_commit=False)

    async with session_factory() as session:
        remaining_tags = (await session.execute(select(Tag))).scalars().all()
        remaining_categories = (await session.execute(select(Category))).scalars().all()

    assert remaining_tags == []
    assert remaining_categories == []


@pytest.mark.asyncio
async def test_add_and_remove_disabled_categories(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    set_id = "set-disabled"
    await semantic_config_storage.set_setid_config(set_id=set_id)

    await semantic_config_storage.add_disabled_category_to_setid(
        set_id=set_id,
        category_name="default-profile",
    )
    await semantic_config_storage.add_disabled_category_to_setid(
        set_id=set_id,
        category_name="default-history",
    )
    # duplicate should be a no-op
    await semantic_config_storage.add_disabled_category_to_setid(
        set_id=set_id,
        category_name="default-profile",
    )

    config_with_disabled = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert sorted(config_with_disabled.disabled_categories or []) == [
        "default-history",
        "default-profile",
    ]

    await semantic_config_storage.remove_disabled_category_from_setid(
        set_id=set_id,
        category_name="default-profile",
    )

    config_after_removal = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert config_after_removal.disabled_categories == ["default-history"]


@pytest.mark.asyncio
async def test_org_set_categories_are_inherited_by_mapped_set_id(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-a",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    inherited_category_id = await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="org-category",
        prompt="Org prompt",
    )
    await semantic_config_storage.add_tag(
        category_id=inherited_category_id,
        tag_name="t1",
        description="Tag 1",
    )

    set_id = "set-inherits"
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id,
        org_set_id=org_set_id,
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert [c.name for c in config.categories] == ["org-category"]
    assert config.categories[0].origin_type == "org_tag_set"
    assert config.categories[0].origin_id == org_set_id
    assert config.categories[0].inherited is True
    assert cast(StructuredSemanticPrompt, config.categories[0].prompt).tags == {
        "t1": "Tag 1"
    }


@pytest.mark.asyncio
async def test_set_local_category_overrides_inherited_by_name(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-b",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="conflict",
        prompt="Inherited prompt",
    )

    set_id = "set-overrides"
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id,
        org_set_id=org_set_id,
    )

    await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="conflict",
        prompt="Local prompt",
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)

    assert [c.name for c in config.categories] == ["conflict"]
    assert config.categories[0].origin_type == "set_id"
    assert config.categories[0].origin_id == set_id
    assert config.categories[0].inherited is False


@pytest.mark.asyncio
async def test_register_set_id_org_set_after_setid_config(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-c",
        org_level_set=False,
        metadata_tags=["repo"],
    )
    await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="org-category",
        prompt="Org prompt",
    )

    set_id = "set-has-config"
    await semantic_config_storage.set_setid_config(
        set_id=set_id,
        embedder_name="embedder",
        llm_name="llm",
    )

    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id,
        org_set_id=org_set_id,
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert [c.name for c in config.categories] == ["org-category"]


@pytest.mark.asyncio
async def test_register_set_id_org_set_is_first_write_wins(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    org_set_a = await semantic_config_storage.add_org_set_id(
        org_id="org-d",
        org_level_set=False,
        metadata_tags=["repo"],
    )
    org_set_b = await semantic_config_storage.add_org_set_id(
        org_id="org-d",
        org_level_set=False,
        metadata_tags=["repo", "other"],
    )

    await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_a,
        category_name="from-a",
        prompt="A",
    )
    await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_b,
        category_name="from-b",
        prompt="B",
    )

    set_id = "set-first-write-wins"
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id,
        org_set_id=org_set_a,
    )
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id,
        org_set_id=org_set_b,
    )

    config = await semantic_config_storage.get_setid_config(set_id=set_id)
    assert [c.name for c in config.categories] == ["from-a"]


@pytest.mark.asyncio
async def test_get_category_set_ids_for_set_id_category(
    semantic_config_storage: SemanticConfigStorage,
):
    """Test that get_category_set_ids returns the single set_id for a set-specific category."""
    set_id = "set-get-ids-1"
    category_id = await semantic_config_storage.create_category(
        set_id=set_id,
        category_name="local-category",
        prompt="Local category prompt",
    )

    set_ids = await semantic_config_storage.get_category_set_ids(
        category_id=category_id,
    )

    assert set_ids == [set_id]


@pytest.mark.asyncio
async def test_get_category_set_ids_for_non_existent_category(
    semantic_config_storage: SemanticConfigStorage,
):
    """Test that get_category_set_ids returns empty list for non-existent category."""
    set_ids = await semantic_config_storage.get_category_set_ids(category_id="999999")

    assert set_ids == []


@pytest.mark.asyncio
async def test_get_category_set_ids_for_org_set_category_no_overrides(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    """Test that get_category_set_ids returns all inheriting set_ids when no overrides exist."""
    # Create org set with a category
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-get-ids-1",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    category_id = await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="shared-category",
        prompt="Shared org category",
    )

    # Register three set_ids to this org set
    set_id_1 = "set-inherit-1"
    set_id_2 = "set-inherit-2"
    set_id_3 = "set-inherit-3"

    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id_1,
        org_set_id=org_set_id,
    )
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id_2,
        org_set_id=org_set_id,
    )
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id_3,
        org_set_id=org_set_id,
    )

    # Get set_ids associated with the org category
    set_ids = await semantic_config_storage.get_category_set_ids(
        category_id=category_id,
    )

    # All three set_ids should be returned since none override the category
    assert sorted(set_ids) == sorted([set_id_1, set_id_2, set_id_3])


@pytest.mark.asyncio
async def test_get_category_set_ids_for_org_set_category_with_overrides(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    """Test that get_category_set_ids excludes set_ids that override the category."""
    # Create org set with a category
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-get-ids-2",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    category_id = await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="overridable-category",
        prompt="Original org prompt",
    )

    # Register four set_ids to this org set
    set_id_1 = "set-no-override-1"
    set_id_2 = "set-override-1"
    set_id_3 = "set-no-override-2"
    set_id_4 = "set-override-2"

    for sid in [set_id_1, set_id_2, set_id_3, set_id_4]:
        await semantic_config_storage.register_set_id_org_set(
            set_id=sid,
            org_set_id=org_set_id,
        )

    # Override the category in two of the set_ids
    await semantic_config_storage.create_category(
        set_id=set_id_2,
        category_name="overridable-category",
        prompt="Override prompt for set 2",
    )
    await semantic_config_storage.create_category(
        set_id=set_id_4,
        category_name="overridable-category",
        prompt="Override prompt for set 4",
    )

    # Get set_ids associated with the org category
    set_ids = await semantic_config_storage.get_category_set_ids(
        category_id=category_id,
    )

    # Only the non-overriding set_ids should be returned
    assert sorted(set_ids) == sorted([set_id_1, set_id_3])

    # Verify that the overriding set_ids do have the category (but local version)
    config_2 = await semantic_config_storage.get_setid_config(set_id=set_id_2)
    assert len(config_2.categories) == 1
    assert config_2.categories[0].name == "overridable-category"
    assert config_2.categories[0].inherited is False


@pytest.mark.asyncio
async def test_get_category_set_ids_for_org_set_category_with_different_category(
    semantic_config_storage: SemanticConfigStorageSqlAlchemy,
):
    """Test that set_ids with different category names still inherit the org category."""
    # Create org set with a category
    org_set_id = await semantic_config_storage.add_org_set_id(
        org_id="org-get-ids-3",
        org_level_set=False,
        metadata_tags=["repo"],
    )

    category_id = await semantic_config_storage.create_org_set_category(
        org_set_id=org_set_id,
        category_name="org-category",
        prompt="Org category prompt",
    )

    # Register two set_ids
    set_id_1 = "set-different-cat-1"
    set_id_2 = "set-different-cat-2"

    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id_1,
        org_set_id=org_set_id,
    )
    await semantic_config_storage.register_set_id_org_set(
        set_id=set_id_2,
        org_set_id=org_set_id,
    )

    # Add a different category to set_id_2 (not an override, different name)
    await semantic_config_storage.create_category(
        set_id=set_id_2,
        category_name="different-category",
        prompt="Different category prompt",
    )

    # Get set_ids associated with the org category
    set_ids = await semantic_config_storage.get_category_set_ids(
        category_id=category_id,
    )

    # Both set_ids should be returned since the local category has a different name
    assert sorted(set_ids) == sorted([set_id_1, set_id_2])

    # Verify that set_id_2 has both categories
    config_2 = await semantic_config_storage.get_setid_config(set_id=set_id_2)
    assert len(config_2.categories) == 2
    category_names = {c.name for c in config_2.categories}
    assert category_names == {"org-category", "different-category"}
