from dataclasses import dataclass
from types import ModuleType


@dataclass
class ProfilePrompt:
    update_prompt: str
    consolidation_prompt: str

    @staticmethod
    def load_from_module(prompt_module: ModuleType):
        update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        consolidation_prompt = getattr(prompt_module, "CONSOLIDATION_PROMPT", "")

        return ProfilePrompt(
            update_prompt=update_prompt,
            consolidation_prompt=consolidation_prompt,
        )
