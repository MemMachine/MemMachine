"""Abstract base class for a language model."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")


class LanguageModel(ABC):
    """Abstract base class for a language model."""

    @abstractmethod
    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        """
        Generate a response with structured output parsing.

        Args:
            system_prompt (str | None, optional):
                The system prompt to guide the model's behavior
                (default: None).
            user_prompt (str | None, optional):
                The user prompt containing the main input
                (default: None).
            max_attempts (int, optional):
                The maximum number of attempts to make before giving up
                (default: 1).
            output_format (type[T]):
                The expected output format or schema for parsing the response.
                Implementation-specific (e.g., Pydantic model, JSON schema)
                (default: None).

        Returns:
            Any:
                The parsed response conforming to the specified output_format.

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying language model API.
            ValueError:
                Invalid input or max_attempts.

        """
        raise NotImplementedError

    @abstractmethod
    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        """
        Generate a response based on the provided prompts and tools.

        Args:
            system_prompt (str | None):
                The system prompt to guide the model's behavior
                (default: None).
            user_prompt (str | None):
                The user prompt containing the main input
                (default: None).
            tools (list[dict[str, Any]] | None):
                A list of tools that the model can use in its response, if supported
                (default: None).
            tool_choice (str | dict[str, str] | None):
                Strategy for selecting tools, if supported.
                Can be "auto" for automatic selection,
                "required" for using at least one tool,
                or a specific tool.
                If None, implementation-defined default is used
                (default: None).
            max_attempts (int):
                The maximum number of attempts to make before giving up
                (default: 1).

        Returns:
            tuple[str, Any]:
                A tuple containing the generated response text
                and tool call outputs (if any).

        Raises:
            ExternalServiceAPIError:
                Errors from the underlying embedding API.
            ValueError:
                Invalid input or max_attempts.

        """
        raise NotImplementedError
