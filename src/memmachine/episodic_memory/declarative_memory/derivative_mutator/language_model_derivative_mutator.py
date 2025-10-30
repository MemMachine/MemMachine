"""
A derivative mutator implementation
that rewrites derivatives using a language model.

This can be used to standardize the perspective of derivatives
for improved consistency and searchability.
"""

from uuid import uuid4

from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.language_model.language_model import LanguageModel

from ..data_types import ContentType, Derivative, EpisodeCluster
from .derivative_mutator import DerivativeMutator

DEFAULT_REWRITE_SYSTEM_PROMPT = """
You are an expert in linguistics.
Your task is to rewrite the DERIVATIVE content as an objective observer in the third person.

Guidelines:
- Rewrite the derivative content as an objective observer in the third person.
- Attribute propositional attitudes to the source of the DERIVATIVE content. Do not represent propositional attitudes as facts.
- Resolve anaphoric references using the CONTEXT text when rewriting the DERIVATIVE content.
- Do not include anaphora. Use names for subjects and objects instead of pronouns.
- Retain as much of the original language as possible to capture all nuance. Do not alter sentence structure or order unless necessary.
- Exclude all phatic expressions, except when the DERIVATIVE content is purely phatic.
- If an expression in the DERIVATIVE content requires a response from another participant in an interaction, then the expression is not phatic.
- If an expression in the DERIVATIVE content expresses a propositional attitude, then it is not phatic.
"""

REWRITE_USER_PROMPT_TEMPLATE = """
You are given DERIVATIVE content derived from the CONTEXT text:

<CONTEXT>
{context}
</CONTEXT>

<DERIVATIVE>
{derivative}
</DERIVATIVE>

Output only the rewritten DERIVATIVE content.
"""


class LanguageModelDerivativeMutatorParams(BaseModel):
    """
    Parameters for LanguageModelDerivativeMutator.

    Attributes:
        language_model (LanguageModel):
            LanguageModel instance
            to use for rewriting derivatives.
        rewrite_system_prompt (str):
            System prompt for rewriting derivatives.
            Use DERIVATIVE to reference the derivative content to rewrite
            and CONTEXT to reference the source episode cluster context.
            The default prompt is designed to guide the language model
            to rewrite derivatives in the third-person perspective.
    """

    language_model: InstanceOf[LanguageModel] = Field(
        ...,
        description=("LanguageModel instance to use for rewriting derivatives"),
    )
    rewrite_system_prompt: str = Field(
        DEFAULT_REWRITE_SYSTEM_PROMPT,
        description=(
            "System prompt for rewriting derivatives. "
            "Use DERIVATIVE to reference the derivative content to rewrite "
            "and CONTEXT to reference the source episode cluster context. "
            "The default prompt is designed to guide the language model "
            "to rewrite derivatives in the third-person perspective"
        ),
    )


class LanguageModelDerivativeMutator(DerivativeMutator):
    """
    Derivative mutator that rewrites derivatives
    using a language model.
    """

    def __init__(self, params: LanguageModelDerivativeMutatorParams):
        """
        Initialize a LanguageModelDerivativeMutator
        with the provided parameters.

        Args:
            params (LanguageModelDerivativeMutatorParams):
                Parameters for the LanguageModelDerivativeMutator.
        """
        super().__init__()

        self._language_model = params.language_model
        self._rewrite_system_prompt = params.rewrite_system_prompt

    async def mutate(
        self,
        derivative: Derivative,
        source_episode_cluster: EpisodeCluster,
    ) -> list[Derivative]:
        (
            output_text,
            _,
        ) = await self._language_model.generate_response(
            system_prompt=self._rewrite_system_prompt,
            user_prompt=REWRITE_USER_PROMPT_TEMPLATE.format(
                context="\n".join(
                    episode.content for episode in source_episode_cluster.episodes
                ),
                derivative=derivative.content,
            ),
        )

        rewritten_derivative_content = output_text.strip()
        return [
            Derivative(
                uuid=uuid4(),
                derivative_type=derivative.derivative_type,
                content_type=ContentType.STRING,
                content=rewritten_derivative_content,
                timestamp=derivative.timestamp,
                filterable_properties=(source_episode_cluster.filterable_properties),
                user_metadata=derivative.user_metadata,
            )
        ]
