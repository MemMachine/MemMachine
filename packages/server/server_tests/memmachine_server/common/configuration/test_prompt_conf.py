from memmachine_server.common.configuration import PromptConf
from memmachine_server.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)


def test_prompt_conf_custom_user_categories() -> None:
    conf = PromptConf(default_user_categories=["coding_prompt"])

    defaults = conf.default_semantic_categories
    user_categories = defaults[SemanticSessionManager.SetType.UserSet]

    assert len(user_categories) == 1
    assert user_categories[0].name == "coding_style"
