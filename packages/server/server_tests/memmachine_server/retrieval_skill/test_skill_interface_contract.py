from __future__ import annotations

from memmachine_server.retrieval_skill.common.skill_api import (
    SkillToolBase,
    SkillToolBaseParam,
)


class DemoSkill(SkillToolBase):
    @property
    def skill_name(self) -> str:
        return "DemoSkill"

    @property
    def skill_description(self) -> str:
        return "demo"

    @property
    def accuracy_score(self) -> int:
        return 1

    @property
    def token_cost(self) -> int:
        return 1

    @property
    def time_cost(self) -> int:
        return 1


def test_skill_base_contracts_use_skill_naming() -> None:
    skill = DemoSkill(SkillToolBaseParam())
    assert skill.skill_name == "DemoSkill"
    assert skill.skill_description == "demo"
    assert skill.skill_tools() == []
