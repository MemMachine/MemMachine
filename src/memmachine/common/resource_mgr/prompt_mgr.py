# Load custom prompts from files if specified in the config,
# overriding defaults.
from memmachine.common.configuration import PromptConf


def load_prompt(prompt_config: PromptConf, key: str, default_value: str) -> str:
    """
    Helper function to load a prompt from a file path specified in the
    config.

    Args:
        prompt_config: The dictionary containing prompt file paths.
        key: The key for the specific prompt to load.
        default_value: The default prompt content to use if the file
        is not specified or found.

    Returns:
        The content of the prompt.
    """

    def load_from_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    custom_prompt_path = prompt_config.get(key)
    if custom_prompt_path:
        with open(custom_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return default_value
