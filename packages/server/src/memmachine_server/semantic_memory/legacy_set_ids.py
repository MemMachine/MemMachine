"""Set ids from before semantic sets were registered with an owner."""

import hashlib
import logging
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from itertools import combinations
from math import comb

from .config_store.config_store import SemanticConfigStorage
from .semantic_model import SetIdT, SetTypeEntry
from .storage.storage_base import SemanticStorage

logger = logging.getLogger(__name__)

_KINDS = ("set_type", "project_set", "user_set", "other_set")

# The tags of an id with more readings than this are not searched.
_MAX_TAG_READINGS = 100_000


async def register_legacy_set_ids(
    *,
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
) -> None:
    """
    Register each unregistered set id that names exactly one owner and metadata.

    An id that reads more than one way is resolved by the set types of the
    orgs it could belong to: a reading counts only when its org has a set
    type of its level and tag keys, or any set type for a reading without
    tags. Ids left with more than one reading, or none, stay unregistered and
    are logged; no operation reaches them.
    """
    set_ids = {
        set_id async for set_id in semantic_storage.get_set_ids_starts_with("mem_")
    }
    set_ids.update(
        set_id
        for set_id in await semantic_config_storage.get_configured_set_ids()
        if set_id.startswith("mem_")
    )

    unregistered = set_ids - await _get_registered(semantic_config_storage, set_ids)
    if not unregistered:
        return

    org_set_types: dict[str, list[SetTypeEntry]] = {}

    async def has_set_type(reading: SemanticConfigStorage.Set) -> bool:
        if reading.org_id not in org_set_types:
            org_set_types[
                reading.org_id
            ] = await semantic_config_storage.list_set_type_ids(org_id=reading.org_id)
        set_types = org_set_types[reading.org_id]
        if not reading.set_metadata:
            return bool(set_types)
        return any(
            set_type.is_org_level == (reading.project_id is None)
            and set(set_type.tags) == set(reading.set_metadata)
            for set_type in set_types
        )

    sets: list[SemanticConfigStorage.Set] = []
    unresolved: list[SetIdT] = []
    for set_id in sorted(unregistered):
        readings = parse_legacy_set_id(set_id)
        if len(readings) > 1:
            readings = [reading for reading in readings if await has_set_type(reading)]
        if len(readings) == 1:
            sets.append(readings[0])
        else:
            unresolved.append(set_id)

    await semantic_config_storage.add_sets(sets=sets)

    registered = await _get_registered(
        semantic_config_storage, [semantic_set.id for semantic_set in sets]
    )
    unresolved.extend(
        semantic_set.id for semantic_set in sets if semantic_set.id not in registered
    )
    logger.info("Registered %d legacy semantic set ids", len(registered))
    if unresolved:
        logger.warning(
            "Left %d legacy semantic set ids unregistered, since each names no "
            "single owner and set metadata, or another set has them. First "
            "ones: %s",
            len(unresolved),
            unresolved[:10],
        )


async def _get_registered(
    semantic_config_storage: SemanticConfigStorage,
    set_ids: Collection[SetIdT],
) -> set[SetIdT]:
    return {
        semantic_set.id
        for semantic_set in await semantic_config_storage.get_sets(set_ids=set_ids)
    }


def parse_legacy_set_id(set_id: SetIdT) -> list[SemanticConfigStorage.Set]:
    """
    Return every owner and set metadata that `legacy_set_id` maps to `set_id`.

    An id with more tag readings than can be searched returns no reading.
    """
    readings: dict[tuple, SemanticConfigStorage.Set] = {}
    for kind in _KINDS:
        prefix = f"mem_{kind}_org_"
        if not set_id.startswith(prefix):
            continue

        rest = set_id[len(prefix) :]
        for match in re.finditer(r"(?=_([0-9]+)_[0-9a-f]{12}__)", rest):
            scope = rest[: match.start()]
            tags = rest[match.start() + len(match[1]) + 16 :]
            for org_id, project_id in _scope_readings(scope):
                for set_metadata in _tag_readings(tags, count=int(match[1])):
                    if (
                        legacy_set_id(
                            org_id=org_id,
                            project_id=project_id,
                            set_metadata=set_metadata,
                        )
                        == set_id
                    ):
                        reading = SemanticConfigStorage.Set(
                            id=set_id,
                            org_id=org_id,
                            project_id=project_id,
                            set_metadata=set_metadata,
                        )
                        key = (org_id, project_id, tuple(sorted(set_metadata.items())))
                        readings[key] = reading
    return list(readings.values())


def _scope_readings(scope: str) -> Iterator[tuple[str, str | None]]:
    if scope:
        yield scope, None
    for match in re.finditer(r"(?=_project_)", scope):
        org_id = scope[: match.start()]
        project_id = scope[match.start() + len("_project_") :]
        if org_id and project_id:
            yield org_id, project_id


def _tag_readings(tags: str, *, count: int) -> Iterator[dict[str, str]]:
    if count == 0:
        if not tags:
            yield {}
        return

    # Each tag is "key_value" and tags are joined by "_", so a reading picks
    # which underscores end a key and which separate tags.
    underscores = [i for i, char in enumerate(tags) if char == "_"]
    cuts = 2 * count - 1
    if comb(len(underscores), cuts) > _MAX_TAG_READINGS:
        return

    for positions in combinations(underscores, cuts):
        bounds = [-1, *positions, len(tags)]
        parts = [tags[bounds[i] + 1 : bounds[i + 1]] for i in range(len(bounds) - 1)]
        keys = parts[0::2]
        if len(set(keys)) == count:
            yield dict(zip(keys, parts[1::2], strict=True))


def legacy_set_id(
    *,
    org_id: str,
    project_id: str | None,
    set_metadata: Mapping[str, str],
) -> SetIdT:
    """Return the id that a set of this org, project and metadata was given."""
    org_project = f"org_{org_id}"
    if project_id is not None:
        org_project = f"{org_project}_project_{project_id}"

    string_tags = [f"{k}_{v}" for k, v in set_metadata.items()]

    if not set_metadata:
        kind = "set_type" if project_id is None else "project_set"
    elif set(set_metadata) == {"producer_id"}:
        kind = "user_set"
    else:
        kind = "other_set"

    return (
        f"mem_{kind}_{org_project}_{len(set_metadata)}"
        f"_{_hash_tag_list(set_metadata.keys())}__{'_'.join(sorted(string_tags))}"
    )


def _hash_tag_list(strings: Iterable[str]) -> str:
    h = hashlib.shake_256()
    for s in sorted(strings):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest(6)
