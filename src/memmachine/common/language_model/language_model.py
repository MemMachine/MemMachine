"""
Abstract base class for a language model.
"""

from abc import ABC, abstractmethod
from typing import Any


class LanguageModel(ABC):
    """
    Abstract base class for a language model.
    """

    @abstractmethod
    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] = "auto",
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """
        Generate a response based on the provided prompts and tools.

        Args:
            system_prompt (str | None, optional):
                The system prompt to guide the model's behavior.
            user_prompt (str | None, optional):
                The user prompt containing the main input.
            tools (list[dict[str, Any]] | None, optional):
                A list of tools that the model can use
                to enhance its response.
            tool_choice (str | dict[str, str], optional):
                Strategy for selecting tools.
                Can be "auto" for automatic selection,
                "required" for using at least one tool,
                or a specific tool.
            max_attempts (int, optional):
                The maximum number of attempts to make before giving up.
                Defaults to 1.


        Returns:
            tuple[str, Any]:
                A tuple containing the generated response text
                and tool call outputs (if any).

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying embedding API.
            ValueError:
                Invalid input or max_attempts.
            RuntimeError:
                Catch-all for any other errors.
        """
        raise NotImplementedError
