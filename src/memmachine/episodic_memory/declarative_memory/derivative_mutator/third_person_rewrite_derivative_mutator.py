"""
A derivative mutator implementation that rewrites derivatives
in the third person using a language model.

This can be used to standardize the perspective of derivatives
for improved consistency and searchability.
"""

from typing import Any
from uuid import uuid4

from memmachine.common.language_model.language_model import LanguageModel
from memmachine.common.utils import isolations_to_session_data

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_mutator import DerivativeMutator


class ThirdPersonRewriteDerivativeMutator(DerivativeMutator):
    """
    Derivative mutator that rewrites derivatives
    in the third person using a language model.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize a ThirdPersonRewriteDerivativeMutator
        with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - language_model (LanguageModel):
                  An instance of a LanguageModel
                  to use for rewriting derivatives.

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        language_model = config.get("language_model")
        if language_model is None:
            raise ValueError("Language model must be provided")
        if not isinstance(language_model, LanguageModel):
            raise TypeError("Language model must be an instance of LanguageModel")

        self._language_model = language_model

    async def mutate(
        self,
        derivative: Derivative,
        source_episode_cluster: EpisodeCluster,
    ) -> list[Derivative]:
        (
            _,
            function_calls_arguments,
        ) = await self._language_model.generate_response(
            system_prompt=THIRD_PERSON_REWRITE_SYSTEM_PROMPT,
            user_prompt=THIRD_PERSON_REWRITE_USER_PROMPT.format(
                context="\n".join(
                    episode.content for episode in source_episode_cluster.episodes
                ),
                derivative=derivative.content,
            ),
            tools=THIRD_PERSON_REWRITE_TOOLS,
            tool_choice={
                "type": "function",
                "name": "submit_rewritten_derivative_content",
            },
            session_data=isolations_to_session_data(
                isolations=source_episode_cluster.filterable_properties,
                default_user_id="",
            ),
        )

        rewritten_derivative_content = [
            function_call_arguments["rewritten_derivative_content"]
            for function_call_arguments in function_calls_arguments
            if "rewritten_derivative_content" in function_call_arguments
        ]

        derivatives = [
            Derivative(
                uuid=uuid4(),
                derivative_type=derivative.derivative_type,
                content_type=ContentType.STRING,
                content=mutated_content,
                timestamp=derivative.timestamp,
                filterable_properties=(source_episode_cluster.filterable_properties),
                user_metadata=derivative.user_metadata,
            )
            for mutated_content in rewritten_derivative_content
        ]
        return derivatives


THIRD_PERSON_REWRITE_SYSTEM_PROMPT = """
You are an expert in linguistics.
"""

THIRD_PERSON_REWRITE_USER_PROMPT = """
You are given DERIVATIVE content derived from the CONTEXT text:

<CONTEXT>
{context}
</CONTEXT>

<DERIVATIVE>
{derivative}
</DERIVATIVE>

Your task is to rewrite the DERIVATIVE content as an objective observer in the third person.

Guidelines:
- Attribute propositional attitudes to the source of the DERIVATIVE content. Do not represent propositional attitudes as facts.
- Resolve anaphoric references using the CONTEXT text when rewriting the DERIVATIVE content.
- Do not include anaphora. Use names for subjects and objects instead of pronouns.
- Retain as much of the original language as possible to capture all nuance. Do not alter sentence structure or order unless necessary.
- Exclude all phatic expressions, except when the DERIVATIVE content is purely phatic.
- If an expression in the DERIVATIVE content requires a response from another participant in an interaction, then the expression is not phatic.
- If an expression in the DERIVATIVE content expresses a propositional attitude, then it is not phatic.
"""

THIRD_PERSON_REWRITE_TOOLS = [
    {
        "type": "function",
        "name": "submit_rewritten_derivative_content",
        "description": "Submit the rewritten derivative content.",
        "parameters": {
            "type": "object",
            "properties": {
                "rewritten_derivative_content": {
                    "type": "string",
                    "description": "The rewritten derivative content.",
                },
            },
            "required": ["rewritten_derivative_content"],
        },
    },
]
