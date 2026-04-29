"""Per-partition configuration for event memory."""

from pydantic import BaseModel, Field


class EventMemoryConf(BaseModel):
    """Per-partition configuration for event memory."""

    embedder: str = Field(
        ...,
        description="Resource ID of the Embedder instance",
    )
    reranker: str | None = Field(
        None,
        description="Resource ID of the Reranker instance. "
        "If None, embedding similarity scores are used for ordering",
    )
    properties_schema: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "User-defined filterable properties and their types. "
            'Maps property name to type name (e.g. {"source_role": "str", "count": "int"}). '
            "Valid types: bool, int, float, str, datetime"
        ),
    )
    derive_sentences: bool = Field(
        False,
        description="Whether to derive sentence-level derivatives from content",
    )
    max_text_chunk_length: int = Field(
        500,
        description="Max code-point length for text chunking in segment creation",
    )
