from memmachine.common.configuration import PromptConf
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager


def test_prompt_conf_includes_user_categories() -> None:
    conf = PromptConf()

    defaults = conf.default_semantic_categories

    user_categories = defaults[SemanticSessionManager.SetType.UserSet]
    org_categories = defaults[SemanticSessionManager.SetType.OrgSet]
    project_categories = defaults[SemanticSessionManager.SetType.ProjectSet]

    assert user_categories == org_categories
    assert user_categories[0].name == "profile"
    assert project_categories[0].name == "profile"


def test_prompt_conf_custom_user_categories() -> None:
    conf = PromptConf(default_user_categories=["coding_prompt"])

    defaults = conf.default_semantic_categories
    user_categories = defaults[SemanticSessionManager.SetType.UserSet]

    assert len(user_categories) == 1
    assert user_categories[0].name == "coding_style"
