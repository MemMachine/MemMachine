from __future__ import annotations

from pathlib import Path

from memmachine_server.common.language_model import materialize_provider_skill_bundle


def test_materialize_provider_skill_bundle_writes_skill_md(tmp_path) -> None:
    bundle = materialize_provider_skill_bundle(
        name="Retrieve Skill",
        description="Top level retrieval",
        skill_markdown="# skill",
        bundle_root=str(tmp_path),
    )

    assert bundle.name == "retrieve-skill"
    assert (Path(bundle.path) / "SKILL.md").exists()


def test_materialize_provider_skill_bundle_is_stable(tmp_path) -> None:
    first = materialize_provider_skill_bundle(
        name="CoQ",
        description="Chain of query",
        skill_markdown="Hello",
        bundle_root=str(tmp_path),
    )
    second = materialize_provider_skill_bundle(
        name="CoQ",
        description="Chain of query",
        skill_markdown="Hello",
        bundle_root=str(tmp_path),
    )

    assert first.path == second.path
