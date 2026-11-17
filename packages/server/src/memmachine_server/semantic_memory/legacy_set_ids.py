"""Set ids from before semantic sets were registered with an owner."""

import hashlib
from collections.abc import Iterable, Mapping

from .semantic_model import SetIdT


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
