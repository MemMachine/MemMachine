"""Per-partition configuration for event memory."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Segmenter configurations.
#
# Each variant is a self-contained Pydantic model with a `type` discriminator.
# Variants that bind to a managed resource (LLM, embedder, etc.) carry the
# resource ID as a `str` field, looked up at instantiation time.


class TextSegmenterConf(BaseModel):
    """Configuration for the recursive-character text segmenter."""

    type: Literal["text"] = "text"
    max_chunk_length: int = Field(
        500,
        description="Max code-point length for text chunks",
    )


SegmenterConfUnion = TextSegmenterConf

SegmenterConf = Annotated[SegmenterConfUnion, Field(discriminator="type")]


# Deriver configurations.


class WholeTextDeriverConf(BaseModel):
    """Configuration for the whole-text deriver."""

    type: Literal["whole_text"] = "whole_text"


class SentenceTextDeriverConf(BaseModel):
    """Configuration for the per-sentence text deriver."""

    type: Literal["sentence_text"] = "sentence_text"


DeriverConfUnion = WholeTextDeriverConf | SentenceTextDeriverConf

DeriverConf = Annotated[DeriverConfUnion, Field(discriminator="type")]


class EventMemoryConf(BaseModel):
    """Per-partition configuration for event memory."""

    vector_store: str = Field(
        ...,
        description=(
            "Resource ID of the vector store backing the derivative index "
            "(e.g. a Qdrant or SQLite vector store under `resources.databases`)"
        ),
    )
    segment_store: str = Field(
        ...,
        description=(
            "Resource ID of the relational database backing the segment store "
            "(e.g. a Postgres or SQLite database under `resources.databases`). "
            "The SegmentStore is created implicitly from this database."
        ),
    )
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
    segmenter: SegmenterConf = Field(
        default_factory=TextSegmenterConf,
        description="Segmenter configuration (discriminated by `type`)",
    )
    deriver: DeriverConf = Field(
        default_factory=WholeTextDeriverConf,
        description="Deriver configuration (discriminated by `type`)",
    )
