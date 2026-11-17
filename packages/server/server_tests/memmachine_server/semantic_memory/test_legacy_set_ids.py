"""Tests for registering set ids from before semantic sets had owners."""

import random
import string

import numpy as np
import pytest

from memmachine_server.semantic_memory.config_store.config_store import (
    SemanticConfigStorage,
)
from memmachine_server.semantic_memory.legacy_set_ids import (
    legacy_set_id,
    parse_legacy_set_id,
    register_legacy_set_ids,
)
from memmachine_server.semantic_memory.semantic_memory import SemanticService
from memmachine_server.semantic_memory.storage.storage_base import SemanticStorage
from server_tests.memmachine_server.semantic_memory.semantic_test_utils import (
    SessionData,
    get_set_id,
)

pytestmark = pytest.mark.asyncio


def _readings(set_id: str) -> set[tuple]:
    return {
        (r.org_id, r.project_id, tuple(sorted(r.set_metadata.items())))
        for r in parse_legacy_set_id(set_id)
    }


async def _add_feature(semantic_storage: SemanticStorage, set_id: str) -> str:
    return await semantic_storage.add_feature(
        set_id=set_id,
        category_name="Profile",
        feature="note",
        value="Alpha note",
        tag="facts",
        embedding=np.array([1.0, -1.0]),
    )


@pytest.mark.parametrize(
    ("org_id", "project_id", "set_metadata"),
    [
        ("acme", None, {}),
        ("acme", "web", {}),
        ("acme", None, {"producer_id": "alice"}),
        ("acme", None, {"team": "eng", "user_id": "u_1"}),
        ("my_org", "my_proj", {}),
    ],
)
async def test_parse_reads_ids_of_plain_names_one_way(org_id, project_id, set_metadata):
    set_id = legacy_set_id(
        org_id=org_id, project_id=project_id, set_metadata=set_metadata
    )

    assert _readings(set_id) == {
        (org_id, project_id, tuple(sorted(set_metadata.items())))
    }


async def test_parse_reads_project_sets_with_tags_two_ways():
    set_id = legacy_set_id(
        org_id="acme", project_id="web", set_metadata={"producer_id": "alice"}
    )

    assert _readings(set_id) == {
        ("acme", "web", (("producer_id", "alice"),)),
        ("acme_project_web", None, (("producer_id", "alice"),)),
    }


@pytest.mark.parametrize("set_id", ["mem_user_alice", "set_a", "mem_set_type_org_"])
async def test_parse_reads_other_ids_no_way(set_id):
    assert parse_legacy_set_id(set_id) == []


async def test_parse_reads_every_id_at_least_its_own_way():
    rng = random.Random(0)
    alphabet = string.ascii_lowercase + string.digits + "_-:"

    def name() -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))

    for _ in range(2000):
        org_id = name()
        project_id = name() if rng.random() < 0.6 else None
        set_metadata = {name(): name() for _ in range(rng.randint(0, 3))}
        set_id = legacy_set_id(
            org_id=org_id, project_id=project_id, set_metadata=set_metadata
        )

        assert (
            org_id,
            project_id,
            tuple(sorted(set_metadata.items())),
        ) in _readings(set_id)


async def test_register_registers_ids_with_one_owner(
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
):
    # Given legacy sets of an org and its project, one of them configured only
    await semantic_config_storage.add_set_type_id(
        org_id="acme", org_level_set=True, metadata_tags=["producer_id"]
    )
    await semantic_config_storage.add_set_type_id(
        org_id="acme", org_level_set=False, metadata_tags=["team"]
    )
    expected = {
        legacy_set_id(org_id="acme", project_id=None, set_metadata={}): (
            "acme",
            None,
            {},
        ),
        legacy_set_id(org_id="acme", project_id="web", set_metadata={}): (
            "acme",
            "web",
            {},
        ),
        legacy_set_id(
            org_id="acme", project_id=None, set_metadata={"producer_id": "alice"}
        ): ("acme", None, {"producer_id": "alice"}),
        legacy_set_id(org_id="acme", project_id="web", set_metadata={"team": "eng"}): (
            "acme",
            "web",
            {"team": "eng"},
        ),
        legacy_set_id(org_id="acme", project_id="my_project_alpha", set_metadata={}): (
            "acme",
            "my_project_alpha",
            {},
        ),
    }
    # The last two ids also read as sets of orgs "acme_project_web" and
    # "acme_project_my", which have no set types
    for set_id in list(expected)[-2:]:
        assert len(parse_legacy_set_id(set_id)) == 2
    configured_only, *with_features = expected
    for set_id in with_features:
        await _add_feature(semantic_storage, set_id)
    await semantic_config_storage.create_category(
        set_id=configured_only, category_name="hobbies", prompt="Hobbies"
    )

    # When registering them
    await register_legacy_set_ids(
        semantic_storage=semantic_storage,
        semantic_config_storage=semantic_config_storage,
    )

    # Then each is registered with its owner and set metadata
    sets = await semantic_config_storage.get_sets(set_ids=list(expected))
    assert {
        s.id: (s.org_id, s.project_id, dict(s.set_metadata)) for s in sets
    } == expected


async def test_register_leaves_ids_of_two_owners_unregistered(
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
    caplog: pytest.LogCaptureFixture,
):
    # Given a set id that both org "acme" with project "web" and org
    # "acme_project_web" could have been given
    await semantic_config_storage.add_set_type_id(
        org_id="acme", org_level_set=False, metadata_tags=["producer_id"]
    )
    await semantic_config_storage.add_set_type_id(
        org_id="acme_project_web", org_level_set=True, metadata_tags=["producer_id"]
    )
    set_id = legacy_set_id(
        org_id="acme", project_id="web", set_metadata={"producer_id": "alice"}
    )
    await _add_feature(semantic_storage, set_id)
    await _add_feature(semantic_storage, "mem_user_alice")

    await register_legacy_set_ids(
        semantic_storage=semantic_storage,
        semantic_config_storage=semantic_config_storage,
    )

    assert (
        await semantic_config_storage.get_sets(set_ids=[set_id, "mem_user_alice"]) == []
    )
    assert "Left 2 legacy semantic set ids unregistered" in caplog.text


async def test_register_skips_ids_already_registered_and_keys_already_taken(
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
):
    registered_id = legacy_set_id(org_id="acme", project_id="web", set_metadata={})
    taken_key_id = legacy_set_id(org_id="acme", project_id=None, set_metadata={})
    await semantic_config_storage.add_sets(
        sets=[
            SemanticConfigStorage.Set(
                id=registered_id, org_id="acme", project_id="web", set_metadata={}
            ),
            SemanticConfigStorage.Set(
                id="new", org_id="acme", project_id=None, set_metadata={}
            ),
        ]
    )
    await _add_feature(semantic_storage, registered_id)
    await _add_feature(semantic_storage, taken_key_id)

    await register_legacy_set_ids(
        semantic_storage=semantic_storage,
        semantic_config_storage=semantic_config_storage,
    )

    assert {
        s.id
        for s in await semantic_config_storage.get_sets(
            set_ids=[registered_id, taken_key_id, "new"]
        )
    } == {registered_id, "new"}


async def test_registered_legacy_sets_keep_their_features(
    semantic_service: SemanticService,
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
):
    session_data = SessionData(org_id="acme", project_id="web")
    set_id = legacy_set_id(org_id="acme", project_id="web", set_metadata={})
    feature_id = await _add_feature(semantic_storage, set_id)

    await register_legacy_set_ids(
        semantic_storage=semantic_storage,
        semantic_config_storage=semantic_config_storage,
    )

    assert await get_set_id(semantic_service, session_data) == set_id
    assert await semantic_service.get_feature(session_data, feature_id) is not None
    assert [
        feature.metadata.id
        async for feature in semantic_service.search(
            message="alpha", session_data=session_data
        )
    ] == [feature_id]
    assert (
        await semantic_service.get_feature(
            SessionData(org_id="acme", project_id="other"), feature_id
        )
        is None
    )
